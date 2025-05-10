import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Define constants
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "company_documents_langchain"
CHROMA_DB_PATH = "../chroma_db_langchain"  # New path for Langchain version
DOC_DIR = ".."  # Relative to this script's location, pointing to the project root

def get_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not found. Please set it.")
        raise ValueError("OpenAI API key not found.")
    return api_key

def main():
    print("Starting Langchain-based embedding process...")

    # Get OpenAI API key
    try:
        openai_api_key = get_openai_api_key()
    except ValueError as e:
        print(f"Failed to get OpenAI API key: {e}")
        print("Please ensure your OPENAI_API_KEY is set as an environment variable.")
        return

    # Define document paths
    doc_files = ["doc1.txt", "doc2.txt"]
    doc_paths = [os.path.join(DOC_DIR, filename) for filename in doc_files]
    
    print(f"Looking for documents in: {os.path.abspath(DOC_DIR)}")
    print(f"Document paths: {doc_paths}")

    # Load documents using Langchain TextLoader
    all_docs = []
    for path in doc_paths:
        try:
            loader = TextLoader(os.path.abspath(path), encoding="utf-8") # TextLoader needs absolute path
            loaded_docs = loader.load()
            # Add source metadata, Langchain loader adds 'source' by default with the full path
            # We can customize if needed, e.g., by setting doc.metadata["filename"] = os.path.basename(path)
            for doc in loaded_docs:
                doc.metadata["filename"] = os.path.basename(path) # Add filename for easier reference
            all_docs.extend(loaded_docs)
            print(f"Loaded document: {os.path.basename(path)}")
        except Exception as e:
            print(f"Error loading document {path} with Langchain: {e}")
    
    if not all_docs:
        print("No documents loaded. Exiting.")
        return
    print(f"Total documents loaded via Langchain: {len(all_docs)}")

    # Split documents into chunks using Langchain TextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(all_docs)
    print(f"Split documents into {len(split_docs)} chunks.")

    if not split_docs:
        print("No chunks generated. Exiting.")
        return

    # Initialize OpenAI embeddings model via Langchain
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=openai_api_key)
        print(f"Initialized OpenAIEmbeddings with model: {EMBEDDING_MODEL}")
    except Exception as e:
        print(f"Error initializing OpenAIEmbeddings: {e}")
        return

    # Initialize ChromaDB vector store via Langchain
    # This will create the directory if it doesn't exist and add documents.
    # The path should be a directory where Chroma can store its files.
    chroma_db_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), CHROMA_DB_PATH))
    print(f"Initializing Chroma vector store at: {chroma_db_full_path}")
    
    try:
        # Chroma.from_documents will create embeddings and store them.
        # It's an efficient way to populate the DB if it's new.
        # For persistent storage, ensure the path is correctly specified.
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=chroma_db_full_path
        )
        vector_store.persist() # Ensure data is written to disk
        print(f"Successfully created/updated Chroma vector store")
        print(f"Total items in collection: {vector_store._collection.count()}")

    except Exception as e:
        print(f"Error creating/updating Chroma vector store with Langchain: {e}")
        return

    print("Langchain-based embedding process completed.")

if __name__ == "__main__":
    # This script is in my_rag_project/my_rag_project/
    # Ensure the script directory exists (it should as part of the project structure)
    main()

