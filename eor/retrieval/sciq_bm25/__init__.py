"""
SciQ BM25 Retriever - retrieves relevant passages from SciQ support corpus using BM25.
"""
import pickle
import os
from ..base import BaseRetriever


class SciQBm25Retriever(BaseRetriever):
    """
    BM25 retriever for SciQ support corpus.
    Retrieves top-k passages from pre-built BM25 index.
    """

    def __init__(self, config):
        super().__init__(config)
        self.bm25_index_path = config.get("bm25_index_path", "./data/datasets/sciq/sciq_bm25_index.pkl")
        self.corpus_path = config.get("corpus_path", "./data/datasets/sciq/support_corpus.json")
        self.top_k = config.get("top_k", 5)

        # Load BM25 index
        if not os.path.exists(self.bm25_index_path):
            raise FileNotFoundError(f"BM25 index not found: {self.bm25_index_path}")

        with open(self.bm25_index_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.corpus = data["corpus"]

        self.verbose_print(f"Loaded BM25 index with {len(self.corpus)} passages")

    def _retrieve(self, query, language, device):
        """
        Retrieve top-k passages for a single query.
        Returns list of passage strings.
        """
        if query in self.result_dict:
            return self.result_dict[query]

        # Tokenize query
        query_tokens = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_k_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]

        # Retrieve passages
        results = [self.corpus[idx]["text"] for idx in top_k_idx]

        self.result_dict[query] = results
        return results

    def _batch_retrieve(self, queries, language_list, device="gpu0"):
        """Batch retrieve passages for multiple queries."""
        results = []
        for query, language in zip(queries, language_list):
            results.append(self._retrieve(query, language, device))
        return results
