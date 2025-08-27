# VidyaSetu — AI Bridge of Knowledge

A small, class-based Retrieval-Augmented Generation (RAG) tutor for conversational QA over documents (example corpus: "THEMES IN WORLD HISTORY" — Class XI).

This repository contains a Python-based pipeline that:

- loads PDF source documents,
- builds or loads a FAISS vector index,
- retrieves relevant passages for a user question,
- forwards context to an LLM (OpenAI Responses API) and returns conversational answers,
- offers a Streamlit UI for interactive usage.

## Table of contents

- [Features](#features)
- [Technology stack](#technology-stack)
- [Repository layout](#repository-layout)
- [Setup](#setup)
  - [Prerequisites](#prerequisites)
  - [Install](#install)
  - [Environment variables](#environment-variables)
  - [Preparing source PDFs](#preparing-source-pdfs)
- [Run](#run)
  - [Run the Streamlit app (recommended)](#run-the-streamlit-app-recommended)
  - [Run interactive CLI tutor](#run-interactive-cli-tutor)
  - [Build vector index only](#build-vector-index-only)
- [Usage](#usage)
- [Configuration & important variables](#configuration--important-variables)
- [Troubleshooting](#troubleshooting)


## Features

- Document ingestion (PDF) using a partitioning approach.
- FAISS vector store for semantic retrieval (load/save to disk).
- OpenAI Responses API integration for conversational answers.
- Streamlit chat UI wrapper (`app.py`).
- Simple CLI chat (`VidyaSetu.py`).

## Technology stack

- Python 3.11+ (tested with 3.12 in this workspace)
- FAISS via `langchain_community.vectorstores.FAISS`
- OpenAI Python client (Responses API) and `OpenAIEmbeddings`
- LangChain (document + retrieval utilities)
- Streamlit for the web UI
- `unstructured` (or an equivalent PDF partitioner) for PDF parsing
- dotenv for environment variable management

See `requirements.txt` for the concrete dependency list used in this project.

## Repository layout (high level)

- `VidyaSetu.py` — main tutor class and CLI chat loop.
- `app.py` — Streamlit web application using `VidyaSetuTutor`.
- `config.py` — project configuration (paths, model names, etc.).
- `requirements.txt` — Python dependencies.
- `faiss_index_from_unstructured/` and `faiss_index_ollama/` — example saved FAISS indexes.
- `test/`, `EC/`, and other helper folders for data and experiments.

## Setup

### Prerequisites

- Python 3.11 or later installed.
- Git (optional, to clone the repo).
- An OpenAI API key with access to the Responses API and embeddings.
- If you plan to parse PDFs locally: the `unstructured` library and its optional system dependencies (check `requirements.txt`).

### Install

1. Clone the repository (if you haven't already) and change into the project directory:

```bash
git clone <repo-url> vidyasetu
cd vidyasetu
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

If you need system dependencies for PDF parsing (e.g., for `unstructured`), follow that project's documentation.

### Environment variables

Create a `.env` file at the project root (same directory as `VidyaSetu.py`) with at least the following variable(s):

```env
OPENAI_API_KEY=sk-...
# Optional, defined in config.py or set here to override
# FAISS_INDEX_PATH=./faiss_index_from_unstructured
# BOOK_SOURCE_DIR=./THEMES IN WORLD HISTORY Textbook for Class XI
# LLM_MODEL=gpt-4o-mini (or your chosen model)
# EMBEDDING_MODEL=text-embedding-3-small
```

Important: keep your OpenAI key secret and never commit it to source control.

### Preparing source PDFs

Place your PDF files in the directory referenced by `config.BOOK_SOURCE_DIR`. By default this project expects a folder with the textbook PDFs (see the repo root for an example folder: `THEMES IN WORLD HISTORY Textbook for Class XI/`). The tutor will create the directory if it does not exist but it must contain the PDFs to build a new vector index.

## Run

### Run the Streamlit app (recommended)

Start the Streamlit UI (uses `app.py`):

```bash
streamlit run app.py
```

This will open a browser UI where you can ask questions. The first run may take longer while the FAISS index is created from your PDFs.

### Run interactive CLI tutor

You can run the CLI chat directly with:

```bash
python VidyaSetu.py
```

This starts a small REPL-style chat where you type questions. It uses the same retrieval + Responses API flow as the Streamlit app.

### Build or rebuild vector index only

If you want to force re-creation of the FAISS index (e.g., after adding new PDFs), delete the existing index directory (value of `config.FAISS_INDEX_PATH`) and rerun either the Streamlit app or `VidyaSetu.py` — the code will detect missing index and create a new one.

Example (replace path with your config value):

```bash
rm -rf ./faiss_index_from_unstructured
python VidyaSetu.py
```

## Usage notes

- The system constructs a retrieval prompt based on the top-k documents from FAISS and sends context to the LLM. Keep questions specific for best answers.
- The Streamlit UI keeps conversation state (previous response ID) across turns using `st.session_state.previous_response_id`.
- The vector store is persisted to disk; re-running the app will reuse the saved index if present.

## Configuration & important variables

Key configuration values live in `config.py`. Typical variables you will see and can customize:

- `BOOK_SOURCE_DIR` — directory that contains source PDFs.
- `FAISS_INDEX_PATH` — where the FAISS index is saved/loaded from.
- `LLM_MODEL` — LLM model name used in the OpenAI Responses API.
- `EMBEDDING_MODEL` — embedding model name used for vectorization.
- `RAG_SYSTEM_PROMPT` — system prompt used to instruct the LLM for RAG answers.

If you prefer environment-driven configuration, you can export these values in `.env` or in your shell before running.

## Troubleshooting

- Error: "OPENAI_API_KEY environment variable not set"
  - Ensure `.env` exists and contains `OPENAI_API_KEY`, or export the variable in your shell.

- PDF parsing errors or empty index
  - Confirm `BOOK_SOURCE_DIR` contains PDF files.
  - Check that system deps required by the PDF partitioner (`unstructured` or similar) are installed.
  - Review printed logs when the vector index is being built for per-file errors.

- Long startup time on index creation
  - Building embeddings and FAISS index can be slow on first run. Use fewer files to test, or pre-build the index on a machine with faster CPU/network.

- Responses are generic or off-topic
  - Make sure the retrieval returned relevant documents (see the Streamlit sidebar, which shows `formatted_docs`).
  - Try rephrasing your question or increasing retrieval `k` in the code where `as_retriever(search_kwargs={"k": 5})` is set.
