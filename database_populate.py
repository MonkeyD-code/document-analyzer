"""
database_populate.py
Creates and saves FAISS (vector) and BM25 (keyword) indexes.
"""

import os
import shutil
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from embeddings_function import get_embedding_function
from data_loader import load_and_split_documents

def create_faiss_and_bm25_indexes(
    data_dir="data",
    faiss_db_path="faiss_index",
    embedding_model="local"
):
    docs = load_and_split_documents(data_dir=data_dir)
    embedding_fn = get_embedding_function(model_name=embedding_model)

    if os.path.exists(faiss_db_path):
        shutil.rmtree(faiss_db_path)
        print(f"Removed existing FAISS database: {faiss_db_path}")

    vectorstore = FAISS.from_documents(docs, embedding_fn)
    vectorstore.save_local(faiss_db_path)
    print(f"FAISS database saved at: {faiss_db_path}")

    tokenized_docs = []
    all_docs_text = []
    for d in docs:
        text = d.page_content
        all_docs_text.append((text, d.metadata))
        tokenized_docs.append(text.split(" "))

    bm25_index = BM25Okapi(tokenized_docs)
    
    return {
        "faiss_path": faiss_db_path,
        "bm25_index": bm25_index,
        "bm25_docs": all_docs_text
    }
