# SOCAnalyst_ChatBot_RAG - SOCMate

## 📌 Problem Statement
SOC analysts work under intense time pressure, needing quick access to vital information hidden in large volumes of technical documents like **security policies, incident response guides, and threat reports**.  

Manually searching these slows investigations, risks missed details, and delays critical decisions.  

To address this challenge, we developed an **intelligent, context-aware chatbot** that leverages **Retrieval-Augmented Generation (RAG)** and locally hosted **Large Language Models (LLMs)** via **Ollama**.  

This system will:
- Ingest domain-specific cybersecurity documents.
- Retrieve the most relevant content in response to user queries.
- Generate concise, accurate answers.

By incorporating **conversational memory**, the chatbot will also understand follow-up questions in the context of prior interactions — enabling faster access to knowledge, reducing cognitive load, and enhancing operational efficiency for SOC analysts.

---

## 🖥️ System Architecture Design
**System Architecture:**  
*(Insert your architecture diagram here)*

---

## ⚙️ Approach
1. Read security PDFs using **pdfplumber** and **PyPDF2**, with OCR (**pytesseract**) for scanned files.
2. Use **spaCy** and **regex** to remove noise and prepare text for semantic search.
3. Break documents into meaningful sections with **LangChain’s SemanticChunker** for accurate retrieval.
4. Create embeddings with **HuggingFace** models and store them in **ChromaDB** for fast search.
5. Improve query accuracy with **MultiQueryRetriever** before sending context to the LLM.
6. Use a local **LLaMA3 model (ChatOllama)** to give precise, context-based answers.
7. Keep chat history using **ConversationBufferMemory** for smooth follow-up questions.
8. Offer a **Gradio** web app so analysts can interact with the assistant effortlessly.

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Data Extraction & OCR
- pdfplumber  
- PyPDF2  
- pytesseract  
- PIL

### Text Processing
- spaCy – NLP preprocessing  
- re – Regex for text cleaning

### Embedding & Chunking
- langchain_huggingface – HuggingFaceEmbeddings  
- langchain_experimental.text_splitter.SemanticChunker – Semantic chunking  
- ChromaDB – Vector database for similarity search

### Retrieval & Generation
- langchain_ollama – Local LLaMA3 integration  
- ConversationalRetrievalChain – RAG pipeline  
- PromptTemplate – Custom prompts  
- MultiQueryRetriever – Query expansion  
- ConversationBufferMemory – Context memory

### Frontend
- gradio

### Utilities
- logging  
- os  
- io  
- sys  
- json
