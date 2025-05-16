# 🚀 GenAI Developer Onboarding Starter Kit

This repo is your all-in-one launchpad for onboarding engineering teams into the world of Generative AI.

Whether you're starting to build internal assistants, integrating Retrieval-Augmented Generation (RAG) into your apps, or scaling GenAI use across departments — this starter kit gives you a hands-on, modular foundation that’s easy to clone, extend, and deploy.

💻 [Start the Hands-On Colab Notebook!](https://colab.research.google.com/github/nnennandukwe/GenAI-Dev-Onboarding-Starter-Kit/blob/main/GenAI_Dev_Onboarding_Starter_Kit.ipynb?utm_source=github&utm_medium=social&utm_campaign=genai_starter_kit_launch)

🗒️ [Fill out this 30-second Feedback Form](https://forms.gle/ztmLsjmUZUtzRQ479?utm_source=colab&utm_medium=notebook&utm_campaign=genai_starter_kit_feedback
)

## 🧠 What's Inside

- ✅ **Interactive RAG pipeline in Colab** using LangChain, OpenAI `gpt-4o` embeddings, and ChromaDB
- 📝 **Markdown-based internal playbook** for async learning and fast ramp-up
- 🧪 **LLM evaluation script** using [Ragas](https://github.com/explodinggradients/ragas) to measure **accuracy**, **completeness**, and **style**
- 💬 **Prompt patterns and templates** to guide better assistant behavior
- 🛠️ **Poetry for dependency management** — no environment headaches

## Overview
- This project demonstrates a simple Retrieval Augmented Generation (RAG) system (where an LLM can query stored documents as extra info to answer prompts)
- It uses OpenAI's `text-embedding-3-small` model to generate embeddings for text documents and stores them in a ChromaDB vector store.
- The system then uses these embeddings to retrieve relevant document chunks based on user queries and generate responses.
- The project also includes an evaluation component using the Ragas framework to assess the performance of the RAG pipeline, focusing on context precision, context recall, and faithfulness.

## Intentions
- To provide a practical example of building a RAG system.
- To showcase the use of OpenAI embeddings for text processing.
- To demonstrate the integration of ChromaDB as a vector store.
- To illustrate how to evaluate a RAG system using Ragas.
- To offer a clear and reproducible setup for others to experiment with RAG systems.

## Get Started with Google Colab Project to run in the browser!
- Go to the [Gen AI Dev Starter Kit Colab](GenAI_Dev_Onboarding_Starter_Kit.ipynb) file to run the code and start building a RAG system!

## Project Structure
```
GenAI-Dev-Onboarding-Starter-Kit/
├──pyproject.toml                         # project dependencies managed by Poetry
├──doc1.txt                               # Example document about "Security and Compliance" 
├──doc2.txt                               # Example document about "Company FAQs"
├──my_rag_project/
    ├── embedding_processor_langchain.py  # Script for generating and storing embeddings
    ├── evaluation_langchain.py           # Script for evaluating the RAG system
└── README.md                             # This file!
```

## Alternative: Local Setup and Installation 

### Prerequisites
- Python 3.9 or higher
- Poetry for dependency management
- An OpenAI API key

### 1. Clone the Repository
If you have cloned this project from a Git repository, navigate to the project directory.

### 2. Install Poetry
If you don't have Poetry installed, you can install it using the following command:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Make sure to add Poetry to your PATH as instructed by the installer.

### 3. Install Dependencies
Navigate to the project's root directory (where `pyproject.toml` is located) and run:
```bash
poetry install
```
This will create a virtual environment and install all necessary packages.

### 4. Set up Environment Variables
This project requires an OpenAI API key. You need to set it as an environment variable named `OPENAI_API_KEY`.

For example, you can add the following line to your shell configuration file (e.g., `.bashrc`, `.zshrc`):
```bash
export OPENAI_API_KEY='your_api_key_here'
```
Then, source the file (e.g., `source ~/.bashrc`) or open a new terminal session.

Alternatively, you can set it directly in your terminal session:
```bash
export OPENAI_API_KEY='your_api_key_here'
```

### 5. Running the Project

#### Step 0: Navigate to `my_rag_project` directory

```bash
cd my_rag_project
```

#### Step 1: Generate Embeddings
The `embedding_processor_langchain.py` script is used to generate embeddings for your documents and store them in ChromaDB. Make sure your documents are in the correct location or **update the script with the correct file path accordingly**!

To run the script:
```bash
poetry run python embedding_processor_langchain.py
```

#### Step 2: Evaluate the RAG System
The `evaluation.py` script is used to evaluate the RAG system. It will use the embeddings generated in the previous step.

To run the script:
```bash
poetry run python evaluation_langchain.py
```

## Troubleshooting & Notes
- Ensure you have a stable internet connection when running the scripts, as they interact with external APIs (OpenAI) and may download data.
- The paths to documents and the ChromaDB database might need adjustment based on where you run the scripts.
- This project is for demonstration purposes and may require further modifications for production use.
