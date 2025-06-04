# Retrieval-Augmented Generation (RAG) using LlamaIndex, FAISS, and OpenAI

This project implements a Retrieval-Augmented Generation (RAG) pipeline that combines **vector-based document retrieval** with **OpenAI's GPT-4 API** to generate contextually accurate responses. The system leverages **LlamaIndex** for document indexing, **FAISS** for fast similarity search, and **Sentence-BERT** embeddings to retrieve the most relevant information from uploaded files.

## 🔍 Objective

To build a domain-adaptive question-answering system that:
- Retrieves semantically relevant chunks from uploaded documents
- Passes them as context to a language model (GPT-4)
- Generates grounded, high-quality answers with reduced hallucination

## 🛠 Technologies and Libraries

- **Python**
- **LlamaIndex** – document indexing and querying
- **FAISS** – vector similarity search
- **OpenAI API** – GPT-4 for response generation
- **SentenceTransformers** – semantic embeddings (e.g., all-MiniLM-L6-v2)
- **Langchain** – prompt orchestration and LLM chaining
- **PyPDF2** – document parsing
- **Streamlit** – front-end interface for user interaction
- **Jupyter Notebook** – prototyping and development

## 🧠 RAG Pipeline Architecture

1. **Document Upload**  
   Users upload PDF documents via a Streamlit UI.

2. **Chunking and Embedding**  
   - Text is extracted using **PyPDF2**
   - Chunked and embedded using **Sentence-BERT**

3. **Indexing with FAISS and LlamaIndex**  
   - Chunks are indexed with **FAISS** for fast nearest-neighbor search
   - Optionally enhanced with **LlamaIndex** query abstraction

4. **Query Handling**  
   - User submits a question
   - Top-k relevant chunks are retrieved based on cosine similarity

5. **Response Generation**  
   - Retrieved context + question are passed to **OpenAI GPT-4**
   - Final answer is generated, grounded in the retrieved content

## 📊 Features

- Document-aware, context-grounded response generation
- Real-time question answering from user-uploaded PDFs
- Streamlit interface for clean interaction
- Modular and extensible pipeline for domain adaptation

## 📈 Example Use Cases

- Academic research assistants
- Legal document Q&A
- Enterprise knowledge base search
- Customer support over internal documentation

## ✅ Future Enhancements

- Add support for DOCX, TXT, and CSV formats  
- Enable vector index persistence and update  
- Fine-tune model prompts using Langchain Templates  
- Add authentication and upload logging  
- Deploy to Hugging Face Spaces or Streamlit Cloud

## ⚠️ Limitations

- FAISS index must be rebuilt on new document upload  
- API usage incurs cost (OpenAI GPT-4)  
- Current implementation assumes English-language input  


