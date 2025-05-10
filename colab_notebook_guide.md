# Colab Notebook Guide: Langchain-Powered RAG System with OpenAI, Chroma, and Ragas

This guide outlines the structure and content for a Colab notebook to run the Langchain-based RAG project.

## 1. Introduction

**Cell Type:** Markdown

```markdown
# Langchain RAG System Demo: OpenAI Embeddings, ChromaDB, and Ragas Evaluation

Welcome! This notebook demonstrates a Retrieval Augmented Generation (RAG) pipeline built using **Langchain**.

**Key steps covered:**
1.  Setting up the environment and installing dependencies (including Langchain).
2.  Processing local documents (`doc1.txt` - Security Guidelines, `doc2.txt` - Company FAQs) using Langchain document loaders and text splitters.
3.  Generating embeddings using Langchain wrappers for OpenAI (`text-embedding-3-small`).
4.  Storing and indexing embeddings in ChromaDB using Langchain integration.
5.  Performing Question Answering using a Langchain `RetrievalQA` chain.
6.  Evaluating the RAG pipeline using Ragas (metrics: context precision, context recall, faithfulness), leveraging Langchain components.

**Before you begin:**
*   You will need an OpenAI API key.
*   The project files (including `embedding_processor_langchain.py`, `evaluation_langchain.py`, `doc1.txt`, `doc2.txt`, and `pyproject.toml`) should be accessible in your Colab environment (e.g., by cloning a Git repository or uploading them).
```

## 2. Setup Environment

**Cell Type:** Code

```python
# Step 2.1: (Optional) Clone the project repository if not already done
# !git clone <your-repo-url>
# %cd <your-repo-name>

# For this example, we assume the files are in the current Colab environment's root or a known path.
# If you uploaded a zip, you might need to unzip it first.
# !unzip my_rag_project.zip # Ensure the zip contains the Langchain updated files
# %cd my_rag_project

print("Environment setup started...")
```

**Cell Type:** Code

```python
# Step 2.2: Install Poetry
!curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH for the current Colab session
import os
os.environ["PATH"] += ":" + os.path.expanduser("~/.local/bin")

!poetry --version
print("Poetry installed.")
```

**Cell Type:** Code

```python
# Step 2.3: Navigate to the project directory
# This should be the directory containing pyproject.toml
# For this guide, we assume the project directory is /content/my_rag_project
import os
project_dir = "/content/my_rag_project"
script_subdir = os.path.join(project_dir, "my_rag_project") # Poetry default src layout

if not os.path.exists(script_subdir):
    os.makedirs(script_subdir)

# Ensure pyproject.toml is present (it would come from your project zip/clone)
# The pyproject.toml should now include langchain, langchain-openai, etc.
# Example of what pyproject.toml should contain (ensure this matches your actual file):
if not os.path.exists(os.path.join(project_dir, "pyproject.toml")):
    with open(os.path.join(project_dir, "pyproject.toml"), "w") as f:
        f.write("""
[tool.poetry]
name = "my-rag-project-colab-langchain"
version = "0.1.0"
description = "RAG with Langchain"
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = ">=3.9,<4.0"
openai = "^1.0.0"
chromadb = "^0.4.0"
ragas = "^0.1.0"
datasets = "^2.0.0"
langchain = "^0.1.0" # Or specific latest version used
langchain-openai = "^0.1.0"
langchain-community = "^0.0.20" # Check versions
langchain-text-splitters = "^0.0.1"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
""")
    print(f"Created dummy pyproject.toml in {project_dir} for Langchain demo. Ensure your actual pyproject.toml is used.")

%cd {project_dir}
print(f"Current directory: {os.getcwd()}")
!ls -la
```

**Cell Type:** Code

```python
# Step 2.4: Install project dependencies using Poetry
# This command reads pyproject.toml and installs all dependencies including Langchain.
!poetry install --no-root
print("Project dependencies (including Langchain) installed.")
```

**Cell Type:** Code

```python
# Step 2.5: Set up OpenAI API Key
import os
from getpass import getpass

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
  api_key = getpass("Please enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = api_key

if os.environ.get("OPENAI_API_KEY"):
    print("OpenAI API key set successfully!")
else:
    print("Failed to set OpenAI API key.")
```

## 3. Data Processing and Embedding (Langchain)

**Cell Type:** Markdown

```markdown
Now, we will use the **Langchain-based script** (`embedding_processor_langchain.py`) to:
1. Load documents (`doc1.txt`, `doc2.txt`) using `TextLoader`.
2. Split documents into chunks using `RecursiveCharacterTextSplitter`.
3. Generate embeddings with `OpenAIEmbeddings` (`text-embedding-3-small`).
4. Store chunks and embeddings in ChromaDB via its Langchain integration.

Make sure `doc1.txt`, `doc2.txt`, and `embedding_processor_langchain.py` (inside the `my_rag_project` subdirectory) are present.
```

**Cell Type:** Code

```python
# Step 3.1: (If not cloned/uploaded) Create dummy document files and the Langchain embedding script
# In a real scenario, these files come from your project.
doc1_content = """## Company Security and Compliance Guidelines... (full content of doc1.txt)"""
doc2_content = """## Company FAQs... (full content of doc2.txt)"""
# Ensure this is the *Langchain version* of the script
embedding_script_langchain_content = """import os... (full content of embedding_processor_langchain.py)"""

project_root = os.getcwd() # Should be /content/my_rag_project
script_dir = os.path.join(project_root, "my_rag_project") # where scripts go

if not os.path.exists(os.path.join(project_root, "doc1.txt")):
    with open(os.path.join(project_root, "doc1.txt"), "w") as f:
        f.write(doc1_content)
    print("Created dummy doc1.txt")

if not os.path.exists(os.path.join(project_root, "doc2.txt")):
    with open(os.path.join(project_root, "doc2.txt"), "w") as f:
        f.write(doc2_content)
    print("Created dummy doc2.txt")

if not os.path.exists(os.path.join(script_dir, "embedding_processor_langchain.py")):
    if not os.path.exists(script_dir):
        os.makedirs(script_dir)
    with open(os.path.join(script_dir, "embedding_processor_langchain.py"), "w") as f:
        f.write(embedding_script_langchain_content)
    print("Created dummy embedding_processor_langchain.py")

!ls -l {project_root}
!ls -l {script_dir}
```

**Cell Type:** Code

```python
# Step 3.2: Run the Langchain embedding processor script
# This script will use Langchain for loading, chunking, embedding, and storing in ChromaDB.
# Ensure your OPENAI_API_KEY is set.
!poetry run python my_rag_project/embedding_processor_langchain.py

print("Langchain embedding process execution attempted.")
# Check for the new chroma_db_langchain directory
!ls -l
!ls -l chroma_db_langchain # This should exist if the script ran successfully
```

## 4. Ragas Evaluation (Langchain Pipeline)

**Cell Type:** Markdown

```markdown
Next, we'll use the **Langchain-based Ragas evaluation script** (`evaluation_langchain.py`). This script will:
1.  Connect to the ChromaDB populated by the Langchain embedding script.
2.  Set up a Langchain `RetrievalQA` chain using an OpenAI LLM (`gpt-3.5-turbo` or similar) and the ChromaDB retriever.
3.  Generate answers for predefined questions using the QA chain.
4.  Retrieve contexts (source documents) used by the QA chain.
5.  Calculate Ragas metrics: `context_precision`, `context_recall`, and `faithfulness`.

Make sure `evaluation_langchain.py` (inside `my_rag_project` subdirectory) is present.
```

**Cell Type:** Code

```python
# Step 4.1: (If not cloned/uploaded) Create dummy Langchain evaluation script
# Ensure this is the *Langchain version* of the script
evaluation_script_langchain_content = """import os... (full content of evaluation_langchain.py)"""

project_root = os.getcwd()
script_dir = os.path.join(project_root, "my_rag_project")

if not os.path.exists(os.path.join(script_dir, "evaluation_langchain.py")):
    if not os.path.exists(script_dir):
        os.makedirs(script_dir)
    with open(os.path.join(script_dir, "evaluation_langchain.py"), "w") as f:
        f.write(evaluation_script_langchain_content)
    print("Created dummy evaluation_langchain.py")

!ls -l {script_dir}
```

**Cell Type:** Code

```python
# Step 4.2: Run the Langchain Ragas evaluation script
# This script requires the ChromaDB (langchain version) to be populated.
# It also requires the OPENAI_API_KEY.
!poetry run python my_rag_project/evaluation_langchain.py

print("Langchain Ragas evaluation process execution attempted.")
```

## 5. Conclusion

**Cell Type:** Markdown

```markdown
This notebook demonstrated the core steps of setting up a RAG pipeline **using Langchain**, from document processing and embedding to QA and evaluation with Ragas.

**Further exploration:**
*   Experiment with different Langchain document loaders, text splitters, and retrievers.
*   Try different LLMs available through Langchain.
*   Explore more advanced Langchain chains and agents for RAG.
*   Expand the evaluation dataset and explore other Ragas metrics in conjunction with Langchain.
```


