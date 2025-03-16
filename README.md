# AI-Powered-Conversational-Knowledge-Retrieval-System
This project is an AI-driven chatbot that uses retrieval-augmented generation (RAG) to answer queries based on stored documents. Built using LangChain, Ollama, ChromaDB, and DeepSeek 7B, it provides context-aware responses with structured memory for improved long-term retention.

**🔹 Features**

**Retrieval-Augmented Generation (RAG)** – Uses ChromaDB to fetch relevant documents before generating responses.

**Conversational Memory** – Retains chat history using LangChain’s Buffer Memory, ensuring contextual continuity.

**DeepSeek 7B LLM**– Generates human-like responses with optimized reasoning.

**Fast & Efficient Retrieval** – Utilizes ChromaDB for high-speed vector search of stored documents.

**Python-Based Implementation**– Fully developed in Python, enabling local inference without cloud dependencies.


**🛠️ Tech Stack**
**Language Model:** DeepSeek 7B via Ollama

**Embedding Model:** Sentence-Transformers (MiniLM L6-v2)

**Vector Database:** ChromaDB

**Memory Management:** LangChain's Conversation Buffer

**🚀 How It Works**

**1️⃣ Load Documents**

Documents are vectorized using HuggingFace embeddings and stored in ChromaDB.

The retriever fetches the most relevant document chunks based on user queries.

**2️⃣ Memory & Context Handling**

LangChain’s Conversation Buffer Memory stores past interactions.

This allows for context-aware responses instead of isolated answers.

**3️⃣ Query Execution**

The query is sent to the retrieval system, which fetches relevant documents.

DeepSeek 7B then generates an AI response, leveraging the retrieved context.

**4️⃣ AI Response Generation**

The AI formats the response, including:

Generated Answer

Relevant Sources

**🔧 Future Improvements**

🔹 Add multi-user session memory

🔹 Enhance retrieval filtering for better accuracy

🔹 Implement fine-tuning for domain-specific knowledge
