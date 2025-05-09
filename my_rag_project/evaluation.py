import os
import chromadb
from openai import OpenAI
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
)
from datasets import Dataset

# Define constants
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "company_documents"
CHROMA_DB_PATH = "../chroma_db"  # Relative to this script's location

# Placeholder for OpenAI API key retrieval
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not found. Please set it.")
        raise ValueError("OpenAI API key not found.")
    return OpenAI(api_key=api_key)

def get_embeddings(client, texts, model=EMBEDDING_MODEL):
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

def retrieve_contexts(client, collection, question, n_results=3):
    """Retrieves contexts from ChromaDB for a given question."""
    query_embedding = get_embeddings(client, [question])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents"] # We need the document content for context
    )
    return results["documents"][0] if results["documents"] and results["documents"][0] else []

def main():
    print("Starting RAGAS evaluation process...")

    # Initialize OpenAI client
    try:
        openai_client = get_openai_client()
    except ValueError as e:
        print(f"Failed to initialize OpenAI client: {e}")
        print("Please ensure your OPENAI_API_KEY is set as an environment variable.")
        return

    # Initialize ChromaDB client and get collection
    chroma_db_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), CHROMA_DB_PATH))
    print(f"Connecting to ChromaDB at: {chroma_db_full_path}")
    try:
        chroma_client = chromadb.PersistentClient(path=chroma_db_full_path)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        print(f"Successfully connected to ChromaDB collection \'{COLLECTION_NAME}\'.")
        if collection.count() == 0:
            print("Warning: ChromaDB collection is empty. Run embedding_processor.py first.")
            # return # Allow to proceed to show Ragas setup, though metrics will be poor.
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        print("Ensure that embedding_processor.py has been run successfully and the DB path is correct.")
        return

    # Define evaluation dataset (questions and ground truths)
    # These should ideally be more extensive and cover various aspects of your documents.
    eval_questions = [
        "What is the company's policy on password complexity?",
        "How should employees report a security incident?",
        "Does the company offer remote work options?",
        "How can I contact customer support?"
    ]
    eval_ground_truths = [
        ["Passwords must be at least 12 characters long and include a mix of uppercase letters, lowercase letters, numbers, and special symbols. Users should avoid easily guessable information."],
        ["Employees should immediately report security incidents to their manager and the IT Help Desk or Security Operations Center (SOC) through designated channels, providing as much detail as possible."],
        ["Yes, the company offers remote work options for many positions. Specifics should be checked in the job description or discussed with the hiring manager."],
        ["Customer support can be reached by emailing [Support Email Address] or calling [Support Phone Number] during operating hours."]
    ]

    retrieved_contexts_list = []
    # For faithfulness, we need a generated "answer". We'll use concatenated contexts as a proxy.
    generated_answers_list = []

    print("Retrieving contexts for evaluation questions...")
    for question in eval_questions:
        contexts = retrieve_contexts(openai_client, collection, question, n_results=3)
        retrieved_contexts_list.append(contexts)
        # Simple proxy for a generated answer: concatenate retrieved contexts
        generated_answers_list.append("\n".join(contexts) if contexts else "No context retrieved.")
        print(f"Q: {question}\nRetrieved contexts: {len(contexts)}")

    # Prepare data for Ragas evaluation
    # Ragas expects a Hugging Face Dataset object
    data = {
        "question": eval_questions,
        "contexts": retrieved_contexts_list,
        "ground_truth": eval_ground_truths, # Ragas expects 'ground_truth' for context_recall, context_precision
        "answer": generated_answers_list # Ragas expects 'answer' for faithfulness
    }
    dataset = Dataset.from_dict(data)

    print("Running Ragas evaluation...")
    # Define metrics
    # Note: Some Ragas metrics might make calls to an LLM (even for evaluation purposes) 
    # which would use the OpenAI API key configured in Ragas internals if not overridden.
    # Faithfulness, for example, often uses an LLM to check consistency.
    metrics_to_evaluate = [
        context_precision,  # Requires question, ground_truth, contexts
        context_recall,     # Requires ground_truth, contexts
        faithfulness,       # Requires answer, contexts
    ]
    
    # Ensure Ragas uses the specified OpenAI model for its internal LLM calls if any
    # This is a bit indirect; Ragas might use its own defaults or require specific configuration
    # for the LLM it uses for certain metrics. For now, we rely on environment variable for OpenAI.
    # from ragas.llms import LangchainLLMWrapper
    # from langchain_openai import ChatOpenAI
    # ragas_llm = LangchainLLMWrapper(ChatOpenAI(model_name="gpt-3.5-turbo")) # Or gpt-4o if preferred
    # faithfulness.llm = ragas_llm # Example of setting LLM for a specific metric

    try:
        result = evaluate(
            dataset,
            metrics=metrics_to_evaluate,
            # llm=ragas_llm, # Optionally provide a specific LLM for Ragas evaluations
            # embeddings= # Optionally provide specific embeddings if needed by a metric
        )
        print("Ragas Evaluation Results:")
        print(result)

        # You can also access individual scores:
        # print(f"Context Precision: {result['context_precision']}")
        # print(f"Context Recall: {result['context_recall']}")
        # print(f"Faithfulness: {result['faithfulness']}")

    except Exception as e:
        print(f"Error during Ragas evaluation: {e}")
        print("This might be due to API key issues, empty contexts, or Ragas internal errors.")
        print("Ensure your OPENAI_API_KEY is valid and has sufficient quota.")

    print("RAGAS evaluation process completed.")

if __name__ == "__main__":
    main()

