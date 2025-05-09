```markdown
# Project: RAG System with OpenAI and ChromaDB

## Overview
This project demonstrates a simple Retrieval Augmented Generation (RAG) system. It uses OpenAI's `text-embedding-3-small` model to generate embeddings for text documents and stores them in a ChromaDB vector store. The system then uses these embeddings to retrieve relevant document chunks based on user queries and generate responses. The project also includes an evaluation component using the Ragas framework to assess the performance of the RAG pipeline, focusing on context precision, context recall, and faithfulness.

## Intentions
- To provide a practical example of building a RAG system.
- To showcase the use of OpenAI embeddings for text processing.
- To demonstrate the integration of ChromaDB as a vector store.
- To illustrate how to evaluate a RAG system using Ragas.
- To offer a clear and reproducible setup for others to experiment with RAG systems.

## Project Structure
```
my_rag_project/
├── embedding_processor.py  # Script for generating and storing embeddings
├── evaluation_script.py    # Script for evaluating the RAG system (conceptual)
├── requirements.txt        # List of Python packages required
└── README.md               # This file
```

## Setup and Installation

### Prerequisites
- Python 3.9 or higher
- Poetry for dependency management
- An OpenAI API key

### 1. Clone the Repository (if applicable)
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

## Running the Project

### Step 1: Generate Embeddings
The `embedding_processor.py` script is used to generate embeddings for your documents and store them in ChromaDB. Make sure your documents are in the correct location or update the script accordingly.

To run the script:
```bash
poetry run python embedding_processor.py
```

### Step 2: Evaluate the RAG System
The `evaluation.py` script is used to evaluate the RAG system. It will use the embeddings generated in the previous step.

To run the script:
```bash
poetry run python evaluation.py
```

## Using Google Colab

To run this project in Google Colaboratory:

1.  **Upload your files**: Upload `embedding_processor.py`, `evaluation.py`, and any necessary document files (e.g., `doc1.txt`, `doc2.txt`) to your Colab environment.
2.  **Install dependencies**: In a Colab cell, run the following commands to install Poetry and project dependencies:
    ```python
    !curl -sSL https://install.python-poetry.org | python3 -
    !poetry install
    ```
3.  **Set OpenAI API Key**: In a Colab cell, set your OpenAI API key as an environment variable:
    ```python
    import os
    os.environ['OPENAI_API_KEY'] = 'your_api_key_here'
    ```
4.  **Run the scripts**: Execute the `embedding_processor.py` and `evaluation.py` scripts as needed using `!poetry run python <script_name>.py`.

## Notes
- Ensure you have a stable internet connection when running the scripts, as they interact with external APIs (OpenAI) and may download data.
- The paths to documents and the ChromaDB database might need adjustment based on where you run the scripts.
- This project is for demonstration purposes and may require further modifications for production use.
```
