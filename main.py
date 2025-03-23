"""
main.py
Coordinates the entire pipeline:
1. Populate DB (FAISS & BM25)
2. Hybrid search
3. Generate a response with a local LLM (Mistral via Ollama)
4. Ensure 100% PDF-based answers with references
"""

from embeddings_function import get_embedding_function
from database_populate import create_faiss_and_bm25_indexes
from retrieval import hybrid_search
from prompt_template import get_prompt_template  # ✅ Using centralized prompt template

# Import updated Ollama
from langchain_ollama import OllamaLLM


def main():
    # Create / Load indexes with local embeddings
    indexes = create_faiss_and_bm25_indexes(
        data_dir="data",
        faiss_db_path="faiss_index",
        embedding_model="local"
    )
    faiss_db_path = indexes["faiss_path"]
    bm25_index = indexes["bm25_index"]
    bm25_docs = indexes["bm25_docs"]

    # Get local embedding function
    embedding_fn = get_embedding_function("local")

    # Ask user for a query
    query_text = input("Enter your question: ")

    # Perform hybrid search with the query
    results = hybrid_search(
        query=query_text,
        faiss_db_path=faiss_db_path,
        bm25_index=bm25_index,
        bm25_docs=bm25_docs,
        embedding_fn=embedding_fn,
        top_k_faiss=3,
        top_k_bm25=3
    )

    if not results:
        print("\n---- Final Answer ----")
        print("Not found in the provided files.")
        return

    # Format context with references
    context_text = "\n\n".join(
        f"{text}\n[Source: {meta.get('source', 'unknown')}]"
        for text, meta, score in results[:3]
    )

    # ✅ Use centralized prompt template
    prompt_template = get_prompt_template()
    final_prompt = prompt_template.format(context=context_text, question=query_text)

    # Use the local Mistral model via Ollama
    llm = OllamaLLM(model="mistral:latest", temperature=0)
    response = llm.invoke(final_prompt)  # Fixed deprecated __call__()

    # Extract references
    references = "\n".join(f"[{i+1}] {meta.get('source', 'unknown')}" for i, (_, meta, _) in enumerate(results[:3]))

    # Print final answer with references
    print("\n---- Final Answer ----")
    print(response)
    print("\n---- References ----")
    print(references)


if __name__ == "__main__":
    main()
