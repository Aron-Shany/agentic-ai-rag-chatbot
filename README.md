# Agentic AI eBook RAG Chatbot

An AI Engineer interview assignment implementing a Retrieval-Augmented Generation (RAG) chatbot for answering questions from the Agentic AI eBook.

## Project Objective

The goal of this project is to build a question-answering system that retrieves relevant information from the Agentic AI eBook and generates answers based only on the retrieved document context.

The system also provides retrieved context and a confidence score for each response.

## Features

- PDF document ingestion
- Text extraction from the Agentic AI eBook
- Text chunking with overlap
- Retrieval of relevant document content
- Context-grounded question answering
- Confidence/relevance score
- Out-of-scope question handling
- Streamlit user interface
- Python-based implementation

## Architecture

```text
Agentic AI eBook (PDF)
        |
        v
   PDF Loader
        |
        v
 Text Chunking
        |
        v
 Embeddings / Retrieval
        |
        v
 Relevant Context
        |
        v
 Grounded Answer
        |
        v
 Answer + Context + Confidence Score