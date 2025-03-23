# Document Analyzer

A RAG (Retrieval-Augmented Generation) based document analysis tool that processes PDF documents, extracts information, and answers questions about their content using a hybrid search approach combining vector similarity (FAISS) and keyword matching (BM25).

## Features

- PDF document loading and parsing with metadata preservation
- Hybrid search combining vector-based (FAISS) and keyword-based (BM25) retrieval
- Local embedding generation using Hugging Face models
- Context-aware question answering with source attribution
- Support for multiple document formats (PDF, TXT, CSV)
- Chunking strategies for optimal retrieval

## Prerequisites

- Python 3.8+ 
- [Optional] Ollama for local LLM inference
- Git

## Installation

1. Clone the repository
```bash
git clone https://github.com/MonkeyD-code/document-analyzer.git
cd document-analyzer
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. [Optional] For local LLM inference, install Ollama and download the Mistral model:
```bash
# Install Ollama from https://ollama.ai/
ollama pull mistral:latest
```

## Usage

### Basic Usage

1. Place your PDF documents in the `data/` directory
2. Run the database population script:
```bash
python database_populate.py
```
3. Ask questions about your documents:
```bash
python main.py
```

### Advanced Usage

To customize the chunking strategy for different document types:

```python
from data_loader import load_and_split_documents

# Create smaller chunks with less overlap for denser documents
chunks = load_and_split_documents(
    data_dir="data",
    chunk_size=300,  # Smaller chunks
    chunk_overlap=20  # Less overlap
)
```

## Architecture

This project implements a hybrid Retrieval-Augmented Generation (RAG) system:

1. **Document Processing**: PDF documents are loaded and split into manageable chunks
2. **Dual Indexing**: 
   - FAISS vector database for semantic similarity search
   - BM25 for keyword-based retrieval
3. **Hybrid Search**: Combines results from both search methods for better recall
4. **Response Generation**: Uses retrieved document chunks as context for LLM response
5. **Source Attribution**: All responses include references to the source documents

## Project Structure

- `data_loader.py`: Functions for loading and processing documents
- `database_populate.py`: Script to populate the vector and keyword databases
- `embeddings_function.py`: Text embedding functionality using Hugging Face models
- `retrieval.py`: Hybrid search implementation (FAISS + BM25)
- `prompt_template.py`: Templates for generating prompts with proper context
- `main.py`: Main application entry point and orchestration
- `faiss_index/`: Directory containing the FAISS index files
- `data/`: Directory for storing your PDF documents

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 