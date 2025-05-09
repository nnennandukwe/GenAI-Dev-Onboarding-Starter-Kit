# Colab Notebook Guide: RAG System with OpenAI, Chroma, and Ragas

This guide outlines the structure and content for a Colab notebook to run the RAG project.

## 1. Introduction

**Cell Type:** Markdown

```markdown
# RAG System Demo: OpenAI Embeddings, ChromaDB, and Ragas Evaluation

Welcome! This notebook demonstrates a simple Retrieval Augmented Generation (RAG) pipeline.

**Key steps covered:**
1.  Setting up the environment and installing dependencies.
2.  Processing local documents (`doc1.txt` - Security Guidelines, `doc2.txt` - Company FAQs).
3.  Generating embeddings using OpenAI (`text-embedding-3-small`).
4.  Storing and indexing embeddings in ChromaDB.
5.  Evaluating the retrieval and generation (placeholder) using Ragas (metrics: context precision, context recall, faithfulness).

**Before you begin:**
*   You will need an OpenAI API key.
*   The project files (including `embedding_processor.py`, `evaluation.py`, `doc1.txt`, `doc2.txt`, and `pyproject.toml`) should be accessible in your Colab environment (e.g., by cloning a Git repository or uploading them).
```

## 2. Setup Environment

**Cell Type:** Code

```python
# Step 2.1: (Optional) Clone the project repository if not already done
# !git clone <your-repo-url>
# %cd <your-repo-name>

# For this example, we assume the files are in the current Colab environment's root or a known path.
# If you uploaded a zip, you might need to unzip it first.
# !unzip my_rag_project.zip
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
# Step 2.3: Navigate to the project directory (if you cloned/unzipped into one)
# This should be the directory containing pyproject.toml
# For example, if your project is in 'my_rag_project':
# %cd my_rag_project

# If files are in the root of your Colab environment and pyproject.toml is there:
# print("Assuming pyproject.toml is in the current directory or a subdirectory like my_rag_project")
# For this guide, we assume the project directory is /content/my_rag_project
# Create dummy project structure for demonstration if not present
import os
project_dir = "/content/my_rag_project"
script_subdir = os.path.join(project_dir, "my_rag_project") # Poetry creates a nested dir

if not os.path.exists(script_subdir):
    os.makedirs(script_subdir)

# Create a dummy pyproject.toml if it doesn't exist for the demo to proceed
# In a real scenario, this file comes from your project.
if not os.path.exists(os.path.join(project_dir, "pyproject.toml")):
    with open(os.path.join(project_dir, "pyproject.toml"), "w") as f:
        f.write("""
[tool.poetry]
name = "my-rag-project-colab"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]

[tool.poetry.dependencies]
python = ">=3.9,<3.13" # Adjusted to be compatible with Colab's default Python if needed
openai = "^1.0.0"
chromadb = "^0.4.0" # Use versions compatible with each other
ragas = "^0.1.0" # Check for latest compatible versions
datasets = "^2.0.0"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
""")
    print(f"Created dummy pyproject.toml in {project_dir}")

# Navigate to the project directory that contains pyproject.toml
%cd {project_dir}

print(f"Current directory: {os.getcwd()}")
!ls -la
```

**Cell Type:** Code

```python
# Step 2.4: Install project dependencies using Poetry
# This command reads pyproject.toml and installs dependencies.
# It might take a few minutes.
!poetry install --no-root # --no-root if you don't need to install the project itself as a package

print("Project dependencies installed.")
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

## 3. Data Processing and Embedding

**Cell Type:** Markdown

```markdown
Now, we will process the local documents (`doc1.txt` and `doc2.txt`), generate embeddings using OpenAI's `text-embedding-3-small` model, and store them in a ChromaDB vector store.

Make sure `doc1.txt`, `doc2.txt`, and `embedding_processor.py` (inside `my_rag_project` subdirectory) are present in your project directory.
```

**Cell Type:** Code

```python
# Step 3.1: (If not cloned/uploaded) Create dummy document files and embedding script
# In a real scenario, these files come from your project.
doc1_content = """## Company Security and Compliance Guidelines... (content of doc1.txt)"""
doc2_content = """## Company FAQs... (content of doc2.txt)"""
embedding_script_content = """import os... (full content of embedding_processor.py)"""

project_root = os.getcwd() # Should be /content/my_rag_project
script_dir = os.path.join(project_root, "my_rag_project") # where embedding_processor.py goes

if not os.path.exists(os.path.join(project_root, "doc1.txt")):
    with open(os.path.join(project_root, "doc1.txt"), "w") as f:
        f.write(doc1_content) # Replace with actual content if needed for testing
    print("Created dummy doc1.txt")

if not os.path.exists(os.path.join(project_root, "doc2.txt")):
    with open(os.path.join(project_root, "doc2.txt"), "w") as f:
        f.write(doc2_content) # Replace with actual content
    print("Created dummy doc2.txt")

if not os.path.exists(os.path.join(script_dir, "embedding_processor.py")):
    if not os.path.exists(script_dir):
        os.makedirs(script_dir)
    with open(os.path.join(script_dir, "embedding_processor.py"), "w") as f:
        f.write(embedding_script_content) # Replace with actual content
    print("Created dummy embedding_processor.py")

!ls -l {project_root}
!ls -l {script_dir}
```

**Cell Type:** Code

```python
# Step 3.2: Run the embedding processor script
# This script will load docs, chunk them, get embeddings, and store in ChromaDB.
# Ensure your OPENAI_API_KEY is set in the environment from the previous step.

# The script is expected to be in the 'my_rag_project' subdirectory, as per standard Poetry structure.
# Poetry run executes commands within the project's virtual environment.
!poetry run python my_rag_project/embedding_processor.py

print("Embedding process execution attempted.")
# Check for chroma_db directory creation
!ls -l
!ls -l chroma_db # This should exist if the script ran successfully
```

## 4. Ragas Evaluation

**Cell Type:** Markdown

```markdown
Next, we'll evaluate the retrieval component using Ragas. The `evaluation.py` script will:
1.  Load a predefined set of questions and ground truth answers.
2.  Retrieve relevant contexts from ChromaDB for each question.
3.  (Placeholder) Generate answers based on retrieved contexts.
4.  Calculate Ragas metrics: `context_precision`, `context_recall`, and `faithfulness`.

Make sure `evaluation.py` (inside `my_rag_project` subdirectory) is present.
```

**Cell Type:** Code

```python
# Step 4.1: (If not cloned/uploaded) Create dummy evaluation script
evaluation_script_content = """import os... (full content of evaluation.py)"""

project_root = os.getcwd() # Should be /content/my_rag_project
script_dir = os.path.join(project_root, "my_rag_project")

if not os.path.exists(os.path.join(script_dir, "evaluation.py")):
    if not os.path.exists(script_dir):
        os.makedirs(script_dir)
    with open(os.path.join(script_dir, "evaluation.py"), "w") as f:
        f.write(evaluation_script_content) # Replace with actual content
    print("Created dummy evaluation.py")

!ls -l {script_dir}
```

**Cell Type:** Code

```python
# Step 4.2: Run the Ragas evaluation script
# This script requires the ChromaDB to be populated by the embedding_processor.py script.
# It also requires the OPENAI_API_KEY for embedding queries and potentially for Ragas' internal LLM calls.
!poetry run python my_rag_project/evaluation.py

print("Ragas evaluation process execution attempted.")
```

## 5. Conclusion

**Cell Type:** Markdown

```markdown
This notebook demonstrated the core steps of setting up a RAG pipeline, from document processing and embedding to evaluation with Ragas.

**Further exploration:**
*   Experiment with different chunking strategies.
*   Integrate a proper Large Language Model (LLM) for answer generation instead of the placeholder.
*   Expand the evaluation dataset for more robust metrics.
*   Explore other Ragas metrics.
```


