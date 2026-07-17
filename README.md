# RAG Document Chatbot 

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using information present in the documents you provide. Built with a self-correcting retrieval loop orchestrated by LangGraph, it evaluates whether the retrieved context is relevant to the user's question. If not, the system rewrites the query and retries retrieval before generating an answer, rather than blindly trusting the top search results.

## How It Works 

1. Upload a PDF or text file. It's split into overlapping chunks, converted into embeddings, and stored in a local vector database (Chroma).
2. When you ask a question, it's embedded the same way and used to search Chroma for the most relevant document chunks.
3. Claude checks whether the retrieved chunks are actually relevant to the question.
   - If relevant → proceed to generate an answer.
   - If not → the query is rewritten and retrieval is retried (up to 2 times).
4. Claude answers the question using only the retrieved context, and cites which chunks it used. If nothing relevant is found even after retries, it says so honestly instead of guessing.

## Stack 

- **Vector Database:** Chroma
- **Embedding Model:** `Sentence-Transformers` (`all-MiniLM-L6-v2`)
- **Large Language Model (LLM):** Claude (Anthropic API)
- **Workflow Orchestration:** LangGraph
- **Frontend:** Streamlit

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/SamrithiSatish/RAG-Document-Chatbot-.git
cd RAG-Document-Chatbot-
```

### 2. Install the required packages

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
pip install -r requirements.txt
```

> **Windows**
> ```bash
> .venv\Scripts\activate
> ```

### 3. Add your Anthropic API key

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

### 4. Add your documents

Place your PDF or `.txt` files inside the `documents/` folder.

### 5. Build the vector database

```bash
python ingest.py
```

### 6. Run the application

```bash
streamlit run app.py
```

The chatbot will open in your browser, and you can upload additional documents directly from the Streamlit sidebar.

## Project Structure

- **`app.py`** – Streamlit user interface
- **`rag.py`** – LangGraph RAG pipeline
- **`ingest.py`** – Document ingestion and vector database creation
- **`documents/`** – Source documents (PDFs and text files)
- **`requirements.txt`** – Project dependencies

## Why the Retry Loop?

A standard RAG pipeline generates an answer from the first set of retrieved documents, even if they aren't relevant to the user's question. This can lead to inaccurate or misleading responses.

This chatbot first checks whether the retrieved documents are relevant. If they aren't, it automatically rewrites the user's query and performs another search before generating a response. If no relevant information is found after retrying, the chatbot responds that it doesn't have enough information rather than producing a potentially incorrect answer.

## Limitations

- Previously ingested documents remain in the vector database until it is manually cleared.
- The relevance checker is AI-based and may occasionally make incorrect decisions.
- Retrieval performance may decline as the document collection grows.


