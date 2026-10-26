from .web import WebRetriever
# from .wiki import WikiRetriever  # 需要 faiss/pyserini，不用 Wiki 时注释掉
from .gen import GenerationRetriever
from ..utils import truncate_en_doc

# 延迟导入 WikiRetriever，只在真正使用时加载
WikiRetriever = None
