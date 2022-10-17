import logging
from datetime import datetime

now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
logger = logging.getLogger("test")
logger.setLevel(level=logging.INFO)

handler = logging.FileHandler("log/"+now+".log", encoding='utf-8')
handler.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
logger.addHandler(console)
