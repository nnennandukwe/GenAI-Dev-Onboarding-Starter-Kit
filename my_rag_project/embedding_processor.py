import os
import chromadb
from openai import OpenAI

# Configure the OpenAI API key
# In a real application, use environment variables or a config file
# For Colab, this will be handled by user input
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# For local testing, you might need to set it directly if not using Colab's environment
# or if the environment variable isn't set in the shell where this script is run by Poetry.
# Ensure this is handled securely and appropriately for the execution environment.

# Define constants
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "company_documents"
CHROMA_DB_PATH = "../chroma_db" # Relative to this script's location in my_rag_project/my_rag_project
DOC_DIR = ".." # Relative to this script's location, pointing to the project root


def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not found. Please set it.")
        # In a Colab environment, you would prompt the user here or use Colab secrets.
        # For now, this will cause an error if not set when running the script.
        # We will add Colab-specific handling later.
        raise ValueError("OpenAI API key not found.")
    return OpenAI(api_key=api_key)

def load_documents(doc_paths):
    """Loads documents from the given file paths."""
    documents = []
    for path in doc_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                documents.append({"name": os.path.basename(path), "content": f.read()})
        except FileNotFoundError:
            print(f"Error: Document not found at {path}")
        except Exception as e:
            print(f"Error loading document {path}: {e}")
    return documents

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Splits text into overlapping chunks."""
    # A simple character-based chunking strategy
    # More sophisticated methods (e.g., by sentences, paragraphs, or using tiktoken) can be used.
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if end >= len(text):
            break
    return chunks

def get_embeddings(client, texts, model=EMBEDDING_MODEL):
    """Generates embeddings for a list of texts using OpenAI."""
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

def main():
    print("Starting embedding process...")
    
    # Initialize OpenAI client
    try:
        client = get_openai_client()
    except ValueError as e:
        print(f"Failed to initialize OpenAI client: {e}")
        print("Please ensure your OPENAI_API_KEY is set as an environment variable.")
        return

    # Define document paths relative to the project root
    # This script is in my_rag_project/my_rag_project, so ../doc1.txt refers to my_rag_project/doc1.txt
    doc_files = ["doc1.txt", "doc2.txt"]
    doc_paths = [os.path.join(DOC_DIR, filename) for filename in doc_files]
    
    print(f"Looking for documents in: {os.path.abspath(DOC_DIR)}")
    print(f"Document paths: {doc_paths}")

    # Load documents
    raw_documents = load_documents(doc_paths)
    if not raw_documents:
        print("No documents loaded. Exiting.")
        return
    print(f"Loaded {len(raw_documents)} documents.")

    all_chunks = []
    all_metadatas = []
    all_ids = []
    chunk_id_counter = 0

    for doc in raw_documents:
        print(f"Processing document: {doc['name']}")
        text_chunks = chunk_text(doc['content'])
        print(f"Split into {len(text_chunks)} chunks.")
        
        if not text_chunks:
            print(f"No chunks generated for {doc['name']}. Skipping.")
            continue

        # Prepare metadatas and ids for ChromaDB
        for i, chunk in enumerate(text_chunks):
            all_chunks.append(chunk)
            all_metadatas.append({"source": doc['name'], "chunk_index": i})
            all_ids.append(f"{doc['name']}_chunk_{chunk_id_counter}")
            chunk_id_counter += 1
    
    if not all_chunks:
        print("No chunks to embed. Exiting.")
        return

    print(f"Total chunks to embed: {len(all_chunks)}")

    # Generate embeddings (handle potential rate limits by batching if necessary, though Chroma client might handle some of this)
    try:
        print(f"Generating embeddings using model: {EMBEDDING_MODEL}...")
        embeddings = get_embeddings(client, all_chunks)
        print(f"Generated {len(embeddings)} embeddings.")
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return

    # Initialize ChromaDB client and collection
    # This will create the directory if it doesn't exist
    chroma_db_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), CHROMA_DB_PATH))
    print(f"Initializing ChromaDB at: {chroma_db_full_path}")
    try:
        chroma_client = chromadb.PersistentClient(path=chroma_db_full_path)
        # Get or create collection. Using get_or_create_collection is idempotent.
        collection = chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            # metadata={"hnsw:space": "cosine"} # Example: specifying distance function if needed
        )
        print(f"ChromaDB collection '{COLLECTION_NAME}' ready.")
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        return

    # Add embeddings to ChromaDB
    try:
        print("Adding embeddings to ChromaDB...")
        collection.add(
            embeddings=embeddings,
            documents=all_chunks, # Store the text chunks themselves
            metadatas=all_metadatas,
            ids=all_ids
        )
        print(f"Successfully added {collection.count()} items to ChromaDB collection '{COLLECTION_NAME}'.")
    except Exception as e:
        print(f"Error adding embeddings to ChromaDB: {e}")
        return

    print("Embedding process completed.")

if __name__ == "__main__":
    # Create the directory for the script if it doesn't exist
    # This script will be placed in my_rag_project/my_rag_project/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(script_dir):
        os.makedirs(script_dir)
    
    main()

