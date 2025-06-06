#!/usr/bin/env python
# coding: utf-8

# # 📘 RAG with LlamaIndex, FAISS, and OpenAI
# This notebook demonstrates how to build a Retrieval-Augmented Generation (RAG) pipeline using LlamaIndex for document indexing, FAISS for vector search, and OpenAI for response generation.

# In[ ]:


pip install llama-index --upgrade


# In[ ]:


get_ipython().system('pip install llama-index-vector-stores-faiss')


# In[ ]:


get_ipython().system('pip install llama-index-embeddings-huggingface')


# ## 🔧 Step 1: Install Required Libraries

# In[ ]:


get_ipython().system('pip install llama-index faiss-cpu sentence-transformers openai PyPDF2')


# ## 📚 Step 2: Import Libraries

# In[ ]:


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.faiss.base import FaissVectorStore
from llama_index.core.storage.storage_context import StorageContext
import openai
import os


# ## 🔑 Step 3: Set Your OpenAI API Key

# In[ ]:


openai.api_key = "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"  # Replace this with your actual API key


# ## 📁 Step 4: Load Documents

# In[ ]:


import os
from google.colab import files
from llama_index.core import SimpleDirectoryReader

# ✅ Create the folder if it doesn't exist
os.makedirs("docs", exist_ok=True)

# ✅ Upload files to /docs
uploaded = files.upload()
for filename in uploaded:
    with open(os.path.join("docs", filename), "wb") as f:
        f.write(uploaded[filename])

# ✅ Load documents from the folder
reader = SimpleDirectoryReader(input_dir="docs", recursive=True)
documents = reader.load_data()

print(f"{len(documents)} documents loaded.")


# ## 🧠 Step 5: Initialize Embedding and LLM Services

# In[ ]:


from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = OpenAI(model="gpt-3.5-turbo")  # or "gpt-4" if available

# Set global config (replaces ServiceContext)
Settings.embed_model = embed_model
Settings.llm = llm


# ## 🔍 Step 6: Create FAISS Vector Index

# In[ ]:


import faiss
from llama_index.vector_stores.faiss.base import FaissVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex

# Step 1: Set embedding dimension (384 for all-MiniLM-L6-v2)
dimension = 384

# Step 2: Create the FAISS index
faiss_index = faiss.IndexFlatL2(dimension)

# Step 3: Pass the FAISS index to the LlamaIndex wrapper
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Step 4: Build the LlamaIndex vector index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

print("✅ Vector index created using FAISS.")


# ## ❓ Step 7: Ask a Question

# In[ ]:


query_engine = index.as_query_engine()
response = query_engine.query("What are the key insights from the documents?")
print("Answer:", response.response)

