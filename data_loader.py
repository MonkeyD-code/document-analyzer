"""
data_loader.py
Loads and splits documents from a data folder.
Supports TXT, PDF, and CSV files.
"""

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_documents(data_dir: str = "data"):
    """
    Loads documents from the given directory.
    Supports .txt, .pdf, and .csv files.
    Returns a list of Document objects with metadata.
    """
    documents = []
    
    # Loop through each file in the data directory
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        
        if file_name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_name.endswith(".csv"):
            loader = CSVLoader(file_path=file_path)
        else:
            continue
        
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file_name
        documents.extend(docs)
    
    return documents

def load_and_split_documents(
    data_dir: str = "data",
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    """
    Loads documents from a directory and splits them into chunks.
    """
    documents = load_documents(data_dir=data_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(documents)
    return split_docs

if __name__ == "__main__":
    docs = load_and_split_documents(data_dir="data")
    print(f"Loaded and split {len(docs)} document chunks:")
    for idx, doc in enumerate(docs):
        print(f"Chunk {idx+1} from source: {doc.metadata.get('source')}")
