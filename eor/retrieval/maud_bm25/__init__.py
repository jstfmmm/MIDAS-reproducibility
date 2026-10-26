import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eor.retrieval.base import BaseRetriever
from legal_retriever import MAUDRetriever


class MAUDBm25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        # 固定返回 top5，在 _process_retrieved_docs 中根据方法名决定取几个
        self.top_k = 5
        self.max_length = config.get("max_length", 2000)
        self.chunk_size = config.get("chunk_size", 500)  # 每个 passage 的词数
        corpus_dir = config.get("corpus_dir", "./legal/corpus/maud")
        benchmark_file = config.get("benchmark_file", "./legal/benchmarks/maud.json")
        bm25_index_path = config.get("bm25_index_path", "./legal/maud_bm25_index.pkl")
        self._retriever = MAUDRetriever(
            corpus_dir=corpus_dir,
            benchmark_file=benchmark_file,
            bm25_index_path=bm25_index_path,
        )

    def _retrieve(self, query, language, device="gpu0"):
        # 返回 top5 个 passage，每个 passage 作为独立的文档
        # 格式与 Web retriever 一致：返回 list[str]
        try:
            from rank_bm25 import BM25Okapi
            import numpy as np

            if self._retriever.bm25 is None:
                self._retriever._load_bm25_index()

            # BM25 检索 top5
            query_tokens = query.lower().split()
            scores = self._retriever.bm25.get_scores(query_tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]

            # 构造独立的文档列表
            docs = []
            for idx in top_indices:
                doc_name, passage = self._retriever.passages[idx]
                # 截断到指定长度
                words = passage.split()[:self.chunk_size]
                doc_text = f"[{doc_name}]\n{' '.join(words)}"
                docs.append(doc_text)

            return docs if docs else [""]
        except Exception as e:
            print(f"BM25 retrieval error: {e}")
            return [""]
