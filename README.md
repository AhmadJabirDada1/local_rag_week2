# Local RAG Chatbot — Week 2

A simple local Retrieval-Augmented Generation (RAG) chatbot built in Python.

## Tech Stack
- Python
- Ollama
- EmbeddingGemma
- Llama 3.2
- PostgreSQL
- pgvector
- pypdf

## How It Works

```text
PDF
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
PostgreSQL + pgvector
↓
Top-K Retrieval
↓
Llama 3.2
↓
Final Answer
```

## Project Structure

```text
local_rag_week2/
├── data/
├── logs/
├── src/
│   ├── config.py
│   ├── document_pipeline.py
│   ├── vector_pipeline.py
│   └── chatbot.py
├── tests/
├── .env
├── requirements.txt
└── README.md
```

## Main Steps

### 1. Document Pipeline
- Loads the PDF
- Extracts and cleans text
- Splits text into overlapping chunks
- Preserves source and page information

### 2. Embeddings
- Uses `embeddinggemma`
- Converts each chunk into a vector

### 3. Vector Storage and Retrieval
- Stores chunks and vectors in PostgreSQL using pgvector
- Converts the user question into an embedding
- Retrieves the Top-K most similar chunks

### 4. Chatbot
- Sends retrieved chunks to `llama3.2:3b`
- Generates the final answer using document context

## Start the Project

Start PostgreSQL:

```bash
open -a Docker
cd ~/Desktop/local_rag_week1
docker compose up -d postgres
```

Start Ollama:

```bash
open -a Ollama
ollama list
```

Open Week 2:

```bash
cd ~/Desktop/local_rag_week2
source .venv/bin/activate
```

## Run Document Pipeline

```bash
python -m src.document_pipeline
```

## Index the Document

```bash
python -m src.vector_pipeline index --reset
```

## Test Retrieval

```bash
python -m src.vector_pipeline query "What is Retrieval-Augmented Generation?"
```

## Run the Chatbot

```bash
python -m src.chatbot
```

Exit with:

```text
exit
```

or:

```text
quit
```

## Run Tests

```bash
python -m pytest -v
```

## Models

```text
Embedding Model: embeddinggemma
Chat Model: llama3.2:3b
```

## Author

Ahmad Jabir Dada
