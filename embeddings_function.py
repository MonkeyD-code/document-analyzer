"""
embeddings_function.py
Defines a function to get embeddings using a local Hugging Face model.
"""

# Updated import from langchain_community
from langchain_huggingface import HuggingFaceEmbeddings



def get_embedding_function(model_name: str = "local"):
    """
    Returns an embedding function based on the chosen model_name.
    For a local solution, we use HuggingFaceEmbeddings.
    """
    if model_name.lower() == "local":
        # You can choose a different model if desired.
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError("Only 'local' embeddings are supported in this configuration.")
