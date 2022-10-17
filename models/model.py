import torch
import torch.nn as nn
import torch.nn.functional as F
# import numpy
# from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput

from .Attention import Attention, Intermediate, Output, Dim_Four_Attention, masked_softmax
from .data_BIO_loader import sentiment2id, validity2id
from allennlp.nn.util import batched_index_select, batched_span_select
import random
import math

def stage_2_features_generation(bert_feature, attention_mask, spans, span_mask, spans_embedding, spans_aspect_tensor,
                                spans_opinion_tensor=None):
    # 对输入的aspect信息进行处理，去除掉无效的aspect span
    all_span_aspect_tensor = None
    all_span_opinion_tensor = None
    all_bert_embedding = None
    all_attention_mask = None
    all_spans_embedding = None
    all_span_mask = None
    spans_aspect_tensor_spilt = torch.chunk(spans_aspect_tensor, spans_aspect_tensor.shape[0], dim=0)
    for i, spans_aspect_tensor_unspilt in enumerate(spans_aspect_tensor_spilt):
        test = spans_aspect_tensor_unspilt.squeeze(0)
        batch_num = spans_aspect_tensor_unspilt.squeeze(0)[0]
        # mask4span_start = torch.where(span_mask[batch_num, :] == 1, spans[batch_num, :, 0], torch.tensor(-1).type_as(spans))
        span_index_start = torch.where(spans[batch_num, :, 0] == spans_aspect_tensor_unspilt.squeeze()[1],
                                       spans[batch_num, :, 1], torch.tensor(-1).type_as(spans))
        span_index_end = torch.where(span_index_start == spans_aspect_tensor_unspilt.squeeze()[2], span_index_start,
                                     torch.tensor(-1).type_as(spans))
        span_index = torch.nonzero((span_index_end > -1), as_tuple=False).squeeze(0)
        if min(span_index.shape) == 0:
            continue
        if spans_opinion_tensor is not None:
            spans_opinion_tensor_unspilt = spans_opinion_tensor[i,:].unsqueeze(0)
        aspect_span_embedding_unspilt = spans_embedding[batch_num, span_index, :].unsqueeze(0)
        bert_feature_unspilt = bert_feature[batch_num, :, :].unsqueeze(0)
        attention_mask_unspilt = attention_mask[batch_num, :].unsqueeze(0)
        spans_embedding_unspilt = spans_embedding[batch_num, :, :].unsqueeze(0)
        span_mask_unspilt = span_mask[batch_num, :].unsqueeze(0)
        if all_span_aspect_tensor is None:
            if spans_opinion_tensor is not None:
                all_span_opinion_tensor = spans_opinion_tensor_unspilt
            all_span_aspect_tensor = aspect_span_embedding_unspilt
            all_bert_embedding = bert_feature_unspilt
            all_attention_mask = attention_mask_unspilt
            all_spans_embedding = spans_embedding_unspilt
            all_span_mask = span_mask_unspilt
        else:
            if spans_opinion_tensor is not None:
                all_span_opinion_tensor = torch.cat((all_span_opinion_tensor, spans_opinion_tensor_unspilt), dim=0)
            all_span_aspect_tensor = torch.cat((all_span_aspect_tensor, aspect_span_embedding_unspilt), dim=0)
            all_bert_embedding = torch.cat((all_bert_embedding, bert_feature_unspilt), dim=0)
            all_attention_mask = torch.cat((all_attention_mask, attention_mask_unspilt), dim=0)
            all_spans_embedding = torch.cat((all_spans_embedding, spans_embedding_unspilt), dim=0)
            all_span_mask = torch.cat((all_span_mask, span_mask_unspilt), dim=0)
    return all_span_opinion_tensor, all_span_aspect_tensor, all_bert_embedding, all_attention_mask, \
           all_spans_embedding, all_span_mask

class Step_1_module(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Step_1_module, self).__init__()
        self.args = args
        self.intermediate = Intermediate(bert_config)
        self.output = Output(bert_config)

    def forward(self, spans_embedding):
        intermediate_output = self.intermediate(spans_embedding)
        layer_output = self.output(intermediate_output, spans_embedding)
        return layer_output, layer_output


class Step_1(torch.nn.Module):
    def feature_slice(self, features, mask, span_mask, sentence_length):
        cnn_span_generate_list = []
        for j, CNN_generation_model in enumerate(self.CNN_span_generation):
            bert_feature = features.permute(0, 2, 1)
            cnn_result = CNN_generation_model(bert_feature)
            cnn_span_generate_list.append(cnn_result)

        features_sliced_tensor = None
        features_mask_tensor = None
        for i in range(features.shape[0]):
            last_mask = torch.nonzero(mask[i, :])
            features_sliced = features[i,:last_mask.shape[0]][1:-1]
            for j in range(self.args.max_span_length -1):
                if last_mask.shape[0] - 2 > j:
                    # test = cnn_span_generate_list[j].permute(0, 2, 1)
                    cnn_feature = cnn_span_generate_list[j].permute(0, 2, 1)[i, 1:last_mask.shape[0] - (j+2), :]
                    features_sliced = torch.cat((features_sliced, cnn_feature), dim=0)
                else:
                    break
            pad_length = span_mask.shape[1] - features_sliced.shape[0]
            spans_mask_tensor = torch.full([1, features_sliced.shape[0]], 1, dtype=torch.long).to(self.args.device)
            if pad_length > 0:
                pad = torch.full([pad_length, self.args.bert_feature_dim], 0, dtype=torch.long).to(self.args.device)
                features_sliced = torch.cat((features_sliced, pad),dim=0)
                mask_pad = torch.full([1, pad_length], 0, dtype=torch.long).to(self.args.device)
                spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad),dim=1)
            if features_sliced_tensor is None:
                features_sliced_tensor = features_sliced.unsqueeze(0)
                features_mask_tensor = spans_mask_tensor
            else:
                features_sliced_tensor = torch.cat((features_sliced_tensor, features_sliced.unsqueeze(0)), dim=0).to(self.args.device)
                features_mask_tensor = torch.cat((features_mask_tensor, spans_mask_tensor), dim=0).to(self.args.device)

        return features_sliced_tensor, features_mask_tensor

    def __init__(self, args, bert_config):
        super(Step_1, self).__init__()
        self.args = args
        self.bert_config = bert_config
        self.dropout_output = torch.nn.Dropout(args.drop_out)
        if self.args.span_generation == "Start_end":
            # 注意此处最大长度要加1的原因是在无效的span的mask由0表示  和其他的span长度结合在一起
            self.step_1_embedding4width = nn.Embedding(args.max_span_length + 1, args.embedding_dim4width)
            self.step_1_linear4width = nn.Linear(args.embedding_dim4width + args.bert_feature_dim * 2,
                                                 args.bert_feature_dim)
        elif self.args.span_generation == "CNN":
            self.CNN_span_generation = nn.ModuleList(
                [nn.Conv1d(in_channels=args.bert_feature_dim, out_channels=args.bert_feature_dim, kernel_size=i + 2) for
                 i in range(args.max_span_length - 1)])
        elif self.args.span_generation == "ATT":
            self.ATT_attentions = nn.ModuleList(
                [Dim_Four_Block(args, self.bert_config) for _ in range(max(1, args.ATT_SPAN_block_num - 1))])
        elif self.args.span_generation == "SE_ATT":
            self.compess_projection = nn.Sequential(nn.Linear(args.bert_feature_dim, 1), nn.ReLU(), nn.Dropout(args.drop_out))

        if args.related_span_underline:
            self.related_attentions = nn.ModuleList(
                [Pointer_Block(args, self.bert_config) for _ in range(max(1, args.related_span_block_num - 1))])

        self.forward_1_decoders = nn.ModuleList(
            [Step_1_module(args, self.bert_config) for _ in range(max(1, args.block_num - 1))])
        self.sentiment_classification_aspect = nn.Linear(args.bert_feature_dim, len(validity2id) - 2)
        # self.sentiment_classification_aspect = nn.Linear(args.bert_feature_dim, len(sentiment2id))

        self.reverse_1_decoders = nn.ModuleList(
            [Step_1_module(args, self.bert_config) for _ in range(max(1, args.block_num - 1))])
        self.sentiment_classification_opinion = nn.Linear(args.bert_feature_dim, len(validity2id) - 2)
        # self.sentiment_classification_opinion = nn.Linear(args.bert_feature_dim, len(sentiment2id))

    def forward(self, input_bert_features, attention_mask, spans, span_mask, related_spans_tensor, sentence_length):

        spans_embedding, features_mask_tensor = self.span_generator(input_bert_features, attention_mask, spans,
                                                                    span_mask, related_spans_tensor, sentence_length)

        if self.args.related_span_underline:
            # spans_embedding_0 = torch.clone(spans_embedding)
            for related_attention in self.related_attentions:
                related_layer_output, related_intermediate_output = related_attention(spans_embedding,
                                                                                      related_spans_tensor,
                                                                                      spans_embedding)
                spans_embedding = related_layer_output
            # spans_embedding = spans_embedding + spans_embedding_0

        span_embedding_1 = torch.clone(spans_embedding)
        for forward_1_decoder in self.forward_1_decoders:
            forward_layer_output, forward_intermediate_output = forward_1_decoder(span_embedding_1)
            span_embedding_1 = forward_layer_output
        class_logits_aspect = self.sentiment_classification_aspect(span_embedding_1)

        span_embedding_2 = torch.clone(spans_embedding)
        for reverse_1_decoder in self.reverse_1_decoders:
            reverse_layer_output, reverse_intermediate_output = reverse_1_decoder(span_embedding_2)
            span_embedding_2 = reverse_layer_output
        class_logits_opinion = self.sentiment_classification_opinion(span_embedding_2)

        return class_logits_aspect, class_logits_opinion, spans_embedding, span_embedding_1, span_embedding_2, \
               features_mask_tensor

    def span_generator(self, input_bert_features, attention_mask, spans, span_mask, related_spans_tensor,
                       sentence_length):
        bert_feature = self.dropout_output(input_bert_features)
        features_mask_tensor = None
        if self.args.span_generation == "Average" or self.args.span_generation == "Max":
            # 如果使用全部span的bert信息：
            spans_num = spans.shape[1]
            spans_width_start_end = spans[:, :, 0:2].view(spans.size(0), spans_num, -1)
            spans_width_start_end_embedding, spans_width_start_end_mask = batched_span_select(bert_feature,
                                                                                              spans_width_start_end)
            spans_width_start_end_mask = spans_width_start_end_mask.unsqueeze(-1).expand(-1, -1, -1,
                                                                                         self.args.bert_feature_dim)
            spans_width_start_end_embedding = torch.where(spans_width_start_end_mask, spans_width_start_end_embedding,
                                                          torch.tensor(0).type_as(spans_width_start_end_embedding))
            if self.args.span_generation == "Max":
                spans_width_start_end_max = spans_width_start_end_embedding.max(2)
                spans_embedding = spans_width_start_end_max[0]
            else:
                spans_width_start_end_mean = spans_width_start_end_embedding.mean(dim=2, keepdim=True).squeeze(-2)
                spans_embedding = spans_width_start_end_mean
        elif self.args.span_generation == "Start_end":
            # 如果使用span区域大小进行embedding
            spans_start = spans[:, :, 0].view(spans.size(0), -1)
            spans_start_embedding = batched_index_select(bert_feature, spans_start)
            spans_end = spans[:, :, 1].view(spans.size(0), -1)
            spans_end_embedding = batched_index_select(bert_feature, spans_end)

            spans_width = spans[:, :, 2].view(spans.size(0), -1)
            spans_width_embedding = self.step_1_embedding4width(spans_width)
            spans_embedding = torch.cat((spans_start_embedding, spans_width_embedding, spans_end_embedding), dim=-1)  # 预留可修改部分
            # spans_embedding_dict = torch.cat((spans_start_embedding, spans_end_embedding, spans_width_embedding), dim=-1)
            spans_embedding_dict = self.step_1_linear4width(spans_embedding)
            spans_embedding = spans_embedding_dict
        elif self.args.span_generation == "CNN":
            feature_slice, features_mask_tensor = self.feature_slice(bert_feature, attention_mask, span_mask,
                                                                     sentence_length)
            spans_embedding = feature_slice
        elif self.args.span_generation == "ATT":
            spans_width_start_end = spans[:, :, 0:2].view(spans.shape[0], spans.shape[1], -1)
            spans_width_start_end_embedding, spans_width_start_end_mask = batched_span_select(bert_feature,
                                                                                              spans_width_start_end)
            span_sum_embdding = torch.sum(spans_width_start_end_embedding, dim=2).unsqueeze(2)
            for ATT_attention in self.ATT_attentions:
                ATT_layer_output, ATT_intermediate_output = ATT_attention(span_sum_embdding,
                                                                                      spans_width_start_end_mask,
                                                                                      spans_width_start_end_embedding)
                span_sum_embdding = ATT_layer_output
            spans_embedding = span_sum_embdding.squeeze()
        elif self.args.span_generation == "SE_ATT":
            spans_width_start_end = spans[:, :, 0:2].view(spans.shape[0], spans.shape[1], -1)
            spans_width_start_end_embedding, spans_width_start_end_mask = batched_span_select(bert_feature,
                                                                                              spans_width_start_end)
            spans_width_start_end_mask_2 = spans_width_start_end_mask.unsqueeze(-1).expand(-1, -1, -1,
                                                                                         self.args.bert_feature_dim)
            spans_width_start_end_embedding = torch.where(spans_width_start_end_mask_2, spans_width_start_end_embedding,
                                                          torch.tensor(0).type_as(spans_width_start_end_embedding))
            claim_self_att = self.compess_projection(spans_width_start_end_embedding).squeeze()
            claim_self_att = torch.sum(spans_width_start_end_embedding, dim=-1).squeeze()
            claim_rep = masked_softmax(claim_self_att, span_mask, spans_width_start_end_mask).unsqueeze(-1).transpose(2, 3)
            claim_rep = torch.matmul(claim_rep, spans_width_start_end_embedding)
            spans_embedding = claim_rep.squeeze()
        return spans_embedding, features_mask_tensor


class Dim_Four_Block(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Dim_Four_Block, self).__init__()
        self.args = args
        self.forward_attn = Dim_Four_Attention(bert_config)
        self.intermediate = Intermediate(bert_config)
        self.output = Output(bert_config)
    def forward(self, hidden_embedding, masks, encoder_embedding):
        #注意， mask需要和attention中的scores匹配，用来去掉对应的无意义的值
        #对应的score的维度为 (batch_size, num_heads, hidden_dim, encoder_dim)
        masks = (~masks) * -1e9
        attention_masks = masks[:, :, None, None, :]
        cross_attention_output = self.forward_attn(hidden_states=hidden_embedding,
                                                   encoder_hidden_states=encoder_embedding,
                                                   encoder_attention_mask=attention_masks)
        attention_output = cross_attention_output[0]
        attention_result = cross_attention_output[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_result


class Pointer_Block(torch.nn.Module):
    def __init__(self, args, bert_config, mask_for_encoder=True):
        super(Pointer_Block, self).__init__()
        self.args = args
        self.forward_attn = Attention(bert_config)
        self.intermediate = Intermediate(bert_config)
        self.output = Output(bert_config)
        self.mask_for_encoder = mask_for_encoder
    def forward(self, hidden_embedding, masks, encoder_embedding):
        #注意， mask需要和attention中的scores匹配，用来去掉对应的无意义的值
        #对应的score的维度为 (batch_size, num_heads, hidden_dim, encoder_dim)
        masks = (~masks) * -1e9
        if masks.dim() == 3:
            attention_masks = masks[:, None, :, :]
        elif masks.dim() == 2:
            if self.mask_for_encoder:
                attention_masks = masks[:, None, None, :]
            else:
                attention_masks = masks[:, None, :, None]
        if self.mask_for_encoder:
            cross_attention_output = self.forward_attn(hidden_states=hidden_embedding,
                                                       encoder_hidden_states=encoder_embedding,
                                                       encoder_attention_mask=attention_masks)
        else:
            cross_attention_output = self.forward_attn(hidden_states=hidden_embedding,
                                                       encoder_hidden_states=encoder_embedding,
                                                       attention_mask=attention_masks)
        attention_output = cross_attention_output[0]
        attention_result = cross_attention_output[1:]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_result


class Step_2_forward(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Step_2_forward, self).__init__()
        self.args = args
        self.bert_config = bert_config
        self.forward_opinion_decoder = nn.ModuleList(
            [Pointer_Block(args, self.bert_config, mask_for_encoder=False) for _ in range(max(1, args.block_num - 1))])
        self.opinion_docoder2class = nn.Linear(args.bert_feature_dim, len(sentiment2id))

    def forward(self, aspect_spans_embedding, aspect_span_mask, spans_aspect_tensor):
        '''aspect---> opinion 方向'''
        for opinion_decoder_layer in self.forward_opinion_decoder:
            opinion_layer_output, opinion_attention = opinion_decoder_layer(aspect_spans_embedding, aspect_span_mask, spans_aspect_tensor)
            aspect_spans_embedding = opinion_layer_output
        opinion_class_logits = self.opinion_docoder2class(aspect_spans_embedding)
        return opinion_class_logits, opinion_attention


class Step_2_reverse(torch.nn.Module):
    def __init__(self, args, bert_config):
        super(Step_2_reverse, self).__init__()
        self.args = args
        self.bert_config = bert_config
        self.reverse_aspect_decoder = nn.ModuleList(
            [Pointer_Block(args, self.bert_config, mask_for_encoder=False) for _ in range(max(1, args.block_num - 1))])
        self.aspect_docoder2class = nn.Linear(args.bert_feature_dim, len(sentiment2id))

    def forward(self, reverse_spans_embedding, reverse_span_mask, all_reverse_opinion_tensor):
        '''opinion---> aspect 方向'''
        for reverse_aspect_decoder_layer in self.reverse_aspect_decoder:
            aspect_layer_output, aspect_attention = reverse_aspect_decoder_layer(reverse_spans_embedding, reverse_span_mask, all_reverse_opinion_tensor)
            reverse_spans_embedding = aspect_layer_output
        aspect_class_logits = self.aspect_docoder2class(reverse_spans_embedding)
        return aspect_class_logits, aspect_attention



def Loss(gold_aspect_label, pred_aspect_label, gold_opinion_label, pred_opinion_label, spans_mask_tensor, opinion_span_mask_tensor,
         reverse_gold_opinion_label, reverse_pred_opinion_label, reverse_gold_aspect_label, reverse_pred_aspect_label,
         cnn_spans_mask_tensor, reverse_aspect_span_mask_tensor, spans_embedding, related_spans_tensor, args):
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    if cnn_spans_mask_tensor is not None:
        spans_mask_tensor = cnn_spans_mask_tensor

    # Loss正向
    aspect_spans_mask_tensor = spans_mask_tensor.view(-1) == 1
    pred_aspect_label_logits = pred_aspect_label.view(-1, pred_aspect_label.shape[-1])
    gold_aspect_effective_label = torch.where(aspect_spans_mask_tensor, gold_aspect_label.view(-1),
                                              torch.tensor(loss_function.ignore_index).type_as(gold_aspect_label))
    aspect_loss = loss_function(pred_aspect_label_logits, gold_aspect_effective_label)

    opinion_span_mask_tensor = opinion_span_mask_tensor.view(-1) == 1
    pred_opinion_label_logits = pred_opinion_label.view(-1, pred_opinion_label.shape[-1])
    gold_opinion_effective_label = torch.where(opinion_span_mask_tensor, gold_opinion_label.view(-1),
                                               torch.tensor(loss_function.ignore_index).type_as(gold_opinion_label))
    opinion_loss = loss_function(pred_opinion_label_logits, gold_opinion_effective_label)
    as_2_op_loss = aspect_loss + opinion_loss

    # Loss反向
    reverse_opinion_span_mask_tensor = spans_mask_tensor.view(-1) == 1
    reverse_pred_opinion_label_logits = reverse_pred_opinion_label.view(-1, reverse_pred_opinion_label.shape[-1])
    reverse_gold_opinion_effective_label = torch.where(reverse_opinion_span_mask_tensor, reverse_gold_opinion_label.view(-1),
                                              torch.tensor(loss_function.ignore_index).type_as(reverse_gold_opinion_label))
    reverse_opinion_loss = loss_function(reverse_pred_opinion_label_logits, reverse_gold_opinion_effective_label)

    reverse_aspect_span_mask_tensor = reverse_aspect_span_mask_tensor.view(-1) == 1
    reverse_pred_aspect_label_logits = reverse_pred_aspect_label.view(-1, reverse_pred_aspect_label.shape[-1])
    reverse_gold_aspect_effective_label = torch.where(reverse_aspect_span_mask_tensor, reverse_gold_aspect_label.view(-1),
                                               torch.tensor(loss_function.ignore_index).type_as(reverse_gold_aspect_label))
    reverse_aspect_loss = loss_function(reverse_pred_aspect_label_logits, reverse_gold_aspect_effective_label)
    op_2_as_loss = reverse_opinion_loss + reverse_aspect_loss

    if args.kl_loss:
        kl_loss = shape_span_embedding(args, spans_embedding, spans_embedding, related_spans_tensor, spans_mask_tensor)
        # loss = as_2_op_loss + op_2_as_loss + kl_loss
        loss = as_2_op_loss + op_2_as_loss + args.kl_loss_weight * kl_loss
    else:
        loss = as_2_op_loss + op_2_as_loss
        kl_loss = 0
    return loss, args.kl_loss_weight * kl_loss

def shape_span_embedding(args, p, q, pad_mask, span_mask):
    kl_loss = 0
    input_size = p.size()
    assert input_size == q.size()
    for i in range(input_size[0]):
        span_mask_index = torch.nonzero(span_mask[i, :]).squeeze()
        lucky_squence = random.choice(span_mask_index)
        P = p[i, lucky_squence, :]
        mask_index = torch.nonzero(pad_mask[i, lucky_squence, :])
        q_tensor = None
        for idx in mask_index:
            if idx == lucky_squence:
                continue
            if q_tensor is None:
                q_tensor = p[i, idx]
            else:
                q_tensor = torch.cat((q_tensor, p[i, idx]), dim=0)
        if q_tensor is None:
            continue
        expan_P = P.expand_as(q_tensor)
        kl_loss += compute_kl_loss(args, expan_P, q_tensor)
    return kl_loss

def compute_kl_loss(args, p, q, pad_mask=None):
    if args.kl_loss_mode == "KLLoss":
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none")
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none")

        if pad_mask is not None:
            p_loss.masked_fill(pad_mask, 0.)
            q_loss.masked_fill(pad_mask, 0.)
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
        total_loss = math.log(1+5/((p_loss + q_loss) / 2))
    elif args.kl_loss_mode == "JSLoss":
        m = (p+q)/2
        m_loss = 0.5 * F.kl_div(F.log_softmax(p, dim=-1), F.softmax(m, dim=-1), reduction="none") + 0.5 * F.kl_div(
            F.log_softmax(q, dim=-1), F.softmax(m, dim=-1), reduction="none")
        if pad_mask is not None:
            m_loss.masked_fill(pad_mask, 0.)
        m_loss = m_loss.sum()
        # test = -math.log(2*m_loss)-math.log(-2*m_loss+2)
        total_loss = 10*(math.log(1+5/m_loss))
    elif args.kl_loss_mode == "EMLoss":
        test = torch.square(p-q)
        em_loss = torch.sqrt(torch.sum(torch.square(p - q)))
        total_loss = math.log(1+5/(em_loss))
    elif args.kl_loss_mode == "CSLoss":
        test = torch.cosine_similarity(p, q, dim=1)
        cs_loss = torch.sum(torch.cosine_similarity(p, q, dim=1))
        total_loss = math.log(1 + 5 / (cs_loss))
    else:
        total_loss = 0
        print('损失种类错误')
    return  total_loss


if __name__ == '__main__':
    tensor1 = torch.zeros((3,3))
    tensor2 = torch.nonzero(tensor1, as_tuple=False)
    tensor1 = tensor1.type_as(tensor2)
    print('666')
