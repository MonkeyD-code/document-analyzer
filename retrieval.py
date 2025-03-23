"""
retrieval.py
Implements hybrid search (FAISS + BM25) and optional reranking.
"""

import numpy as np
from langchain_community.vectorstores import FAISS

def hybrid_search(
    query: str,
    faiss_db_path: str,
    bm25_index,
    bm25_docs,
    embedding_fn,
    top_k_faiss=3,
    top_k_bm25=3
):
    # FAISS search
    faiss_store = FAISS.load_local(faiss_db_path, embedding_fn, allow_dangerous_deserialization=True)
    faiss_results = faiss_store.similarity_search_with_score(query, k=top_k_faiss)

    # BM25 search
    query_tokens = query.split(" ")
    bm25_scores = bm25_index.get_scores(query_tokens)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k_bm25]

    bm25_results = []
    for i in top_bm25_indices:
        bm25_results.append((bm25_docs[i][0], bm25_docs[i][1], bm25_scores[i]))

    faiss_converted = []
    for doc, score in faiss_results:
        faiss_converted.append((doc.page_content, doc.metadata, score))

    combined = faiss_converted + bm25_results

    final_results = []
    for text, meta, score in combined:
        if score <= 1.0:
            new_score = 1.0 - score
        else:
            new_score = score
        final_results.append((text, meta, new_score))

    final_results.sort(key=lambda x: x[2], reverse=True)
    return final_results
