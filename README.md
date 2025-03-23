# Document Analyzer

A RAG (Retrieval-Augmented Generation) based document analysis tool that processes PDF documents, extracts information, and answers questions about their content.

## Features

- PDF document loading and parsing
- Vector-based document retrieval using FAISS
- Efficient embedding generation
- Question answering based on document content

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/document-analyzer.git
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

## Usage

1. Put your PDF documents in the `data/` directory
2. Run the database population script:
```bash
python database_populate.py
```
3. Ask questions about your documents:
```bash
python main.py
```

## Project Structure

- `data_loader.py`: Functions for loading and processing PDF documents
- `database_populate.py`: Script to populate the vector database
- `embeddings_function.py`: Text embedding functionality
- `retrieval.py`: Document retrieval logic
- `prompt_template.py`: Templates for generating prompts
- `main.py`: Main application entry point
- `faiss_index/`: Directory containing the FAISS index files 