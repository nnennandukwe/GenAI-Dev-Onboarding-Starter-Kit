# 🔍 Welcome to the GenAI Developer Starter Kit

Before you jump into the hands-on lab, here’s a quick primer to ground you in **what you’re building**, **why it matters**, and **how each tool fits in**.

---

## 🧠 What Is RAG (Retrieval-Augmented Generation)?

RAG stands for **Retrieval-Augmented Generation** — a technique that improves LLM responses by injecting relevant context *at runtime* from external sources like documents, databases, or APIs.

Instead of fine-tuning the model, RAG retrieves documents and **adds them to the prompt**, helping the LLM give:
- More accurate answers  
- Context-aware responses  
- Better performance on domain-specific tasks

RAG is essential when:
- Your data is private or changes frequently  
- You want explainability and traceability  
- Fine-tuning is too expensive or brittle

---

## 🧰 Tools We’re Using (And Why)

### ✅ **LangChain**
> The orchestration framework

LangChain lets you chain together LLM calls, retrieval, prompt templates, and post-processing — like building workflows with AI blocks.

You’ll use LangChain to:
- Load documents
- Run a retriever
- Generate answers
- Format prompts for evaluation

---

### ✅ **ChromaDB**
> The vector store

We use Chroma to store and search document embeddings. This lets the retriever fetch relevant chunks based on a user query.

Why Chroma?
- Fast and lightweight
- Local and serverless-friendly
- Plays well with LangChain

---

### ✅ **Ragas**
> The evaluation framework

Ragas helps you **evaluate your GenAI system** — not just run it.

We use it to score:
- **Answer Relevance**
- **Factual Correctness**
- **Style and Clarity**

It uses reference answers to calculate quality metrics, which you can use to:
- Compare model performance
- Debug poor generations
- Choose the best configuration for production

---

### ✅ **Poetry**
> Dependency manager

We use Poetry instead of raw pip to:
- Lock versions
- Avoid dependency conflicts
- Keep everything reproducible

---

## 📦 What You’ll Build

In the hands-on lab, you’ll:
1. Load company docs into Chroma
2. Ask a GenAI question using LangChain
3. Get an answer from your RAG pipeline
4. Score the output using Ragas

---

🛠 Ready to go?  
👉 [Start the Colab Notebook »](https://buff.ly/CdngC1j)

---

