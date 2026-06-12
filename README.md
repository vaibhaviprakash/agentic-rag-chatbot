## Overview

This project is an Agentic Retrieval-Augmented Generation (RAG) chatbot built using Python, LangChain, LangGraph, FAISS, Hugging Face Embeddings, Sentence Transformers, FLAN-T5, and Gradio.

The chatbot loads a PDF document, extracts and processes the text, splits it into smaller chunks, converts the chunks into embeddings, stores them in a FAISS vector database, retrieves relevant information based on user queries, and generates answers using Google's FLAN-T5 language model.

A LangGraph workflow orchestrates the application through routing, retrieval, and generation nodes. The router decides whether a query should use PDF retrieval or direct language model generation.

---

## PDF Knowledge Base

The sample PDF contains introductory concepts related to Artificial Intelligence and Large Language Models. It covers topics such as:

* Artificial Intelligence (AI) and its role in building systems capable of performing tasks that normally require human intelligence.
* Large Language Models (LLMs) and how they are trained on large amounts of text data to understand and generate language.
* Retrieval-Augmented Generation (RAG) and its combination of information retrieval and language generation.
* Reinforcement Learning from Human Feedback (RLHF) and its role in aligning model responses with human preferences.
* Embeddings and how text is converted into numerical vectors that capture semantic meaning.
* LangChain and its use in building applications powered by large language models.
* Vector Databases and their role in storing embeddings and enabling similarity search for RAG systems.

## Features

* PDF document loading and processing
* Text chunking using Recursive Character Text Splitter
* Semantic search using Hugging Face embeddings
* FAISS vector database for similarity search
* Retrieval-Augmented Generation (RAG)
* LangGraph workflow orchestration
* Decision-making router for query routing
* FLAN-T5 based answer generation
* Gradio web interface

---

## Tech Stack

* Python
* LangChain
* LangGraph
* FAISS
* Hugging Face Embeddings
* Sentence Transformers (all-MiniLM-L6-v2)
* FLAN-T5 Base
* Gradio

---

## Installation

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python gradio_app.py
```

---

## Workflow

User Query

↓

Router Node

↓

PDF Retrieval Route or General Generation Route

↓

FAISS Similarity Search

↓

Context Retrieval

↓

FLAN-T5 Generation

↓

Answer

---

## Example Questions

* What is Artificial Intelligence?
* What are Large Language Models?
* What is RAG?
* What is RLHF?
* What are embeddings?
* What is LangChain?
* What is a vector database?

---

## Sample PDF

The repository includes a sample PDF (`sample_txt.pdf`) containing introductory concepts related to Artificial Intelligence, Large Language Models, RAG, RLHF, Embeddings, LangChain, and Vector Databases. This document is used as the knowledge base for retrieval and question answering.



---

## Author

Vaibhavi Prakash
