import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
)
from datasets import Dataset

# Define constants
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo" # Or gpt-4o, ensure it's available and suitable
COLLECTION_NAME = "company_documents_langchain"
CHROMA_DB_PATH = "../chroma_db_langchain"  # Path to the Langchain-populated DB

def get_openai_api_key():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not found. Please set it.")
        raise ValueError("OpenAI API key not found.")
    return api_key

def main():
    print("Starting Langchain-based RAGAS evaluation process...")

    # Get OpenAI API key
    try:
        openai_api_key = get_openai_api_key()
    except ValueError as e:
        print(f"Failed to get OpenAI API key: {e}")
        return

    # Initialize OpenAI embeddings and LLM via Langchain
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=openai_api_key)
        llm = ChatOpenAI(model_name=LLM_MODEL, openai_api_key=openai_api_key, temperature=0)
        print(f"Initialized OpenAIEmbeddings with model: {EMBEDDING_MODEL}")
        print(f"Initialized ChatOpenAI with model: {LLM_MODEL}")
    except Exception as e:
        print(f"Error initializing Langchain OpenAI components: {e}")
        return

    # Connect to existing ChromaDB vector store
    chroma_db_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), CHROMA_DB_PATH))
    print(f"Connecting to Chroma vector store at: {chroma_db_full_path}")
    try:
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings, # Provide the embedding function used by the DB
            persist_directory=chroma_db_full_path
        )
        print(f"Successfully connected to Chroma vector store") 
        if vector_store._collection.count() == 0:
            print("Warning: ChromaDB collection is empty. Run embedding_processor_langchain.py first.")
            # return # Allow to proceed to show Ragas setup, though metrics will be poor.

    except Exception as e:
        print(f"Error connecting to Chroma vector store: {e}")
        print("Ensure that embedding_processor_langchain.py has been run successfully.")
        return

    # Create a Langchain retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks

    # Create a Langchain RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" puts all retrieved docs into the context
        retriever=retriever,
        return_source_documents=True # Important for getting contexts for Ragas
    )
    print("Langchain RetrievalQA chain created.")

    # Define evaluation dataset (questions and ground truths)
    eval_questions = [
        "What is the company\'s policy on password complexity?",
        "How should employees report a security incident?",
        "Does the company offer remote work options?",
        "How can I contact customer support?"
    ]
    eval_ground_truths = [
        "Passwords must be at least 12 characters long and include a mix of uppercase letters, lowercase letters, numbers, and special symbols. Users should avoid easily guessable information.",
        "Employees should immediately report security incidents to their manager and the IT Help Desk or Security Operations Center (SOC) through designated channels, providing as much detail as possible.",
        "Yes, the company offers remote work options for many positions. Specifics should be checked in the job description or discussed with the hiring manager.",
        "Customer support can be reached by emailing [Support Email Address] or calling [Support Phone Number] during operating hours."
    ]

    generated_answers_list = []
    retrieved_contexts_list = []

    print("Generating answers and retrieving contexts using Langchain QA chain...")
    for question in eval_questions:
        try:
            response = qa_chain.invoke({"query": question})
            answer = response.get("result", "Failed to generate answer.")
            source_documents = response.get("source_documents", [])
            contexts = [doc.page_content for doc in source_documents]
            
            generated_answers_list.append(answer)
            retrieved_contexts_list.append(contexts)
            print(f"Q: {question}\nA: {answer}\nRetrieved contexts: {len(contexts)}")
        except Exception as e:
            print(f"Error during QA chain invocation for question") 
            generated_answers_list.append("Error in generation.")
            retrieved_contexts_list.append([])

    # Prepare data for Ragas evaluation
    data = {
        "question": eval_questions,
        "contexts": retrieved_contexts_list,
        "ground_truth": eval_ground_truths,
        "answer": generated_answers_list
    }
    dataset = Dataset.from_dict(data)

    print("Running Ragas evaluation...")
    metrics_to_evaluate = [
        context_precision, 
        context_recall,    
        faithfulness,      
    ]

    # Ragas uses OpenAI models by default for some metrics if not configured otherwise.
    # Ensure OPENAI_API_KEY is available in the environment for Ragas.
    # For more control, you can configure Ragas llms and embeddings:
    # from ragas.llms import LangchainLLMWrapper
    # from ragas.embeddings import LangchainEmbeddings
    # ragas_llm = LangchainLLMWrapper(llm) # Use the same LLM as your QA chain
    # ragas_embeddings = LangchainEmbeddings(embeddings) # Use the same embeddings

    try:
        result = evaluate(
            dataset,
            metrics=metrics_to_evaluate,
            # llm=ragas_llm, # if you configured ragas_llm
            # embeddings=ragas_embeddings # if you configured ragas_embeddings
        )
        print("Ragas Evaluation Results (Langchain pipeline):")
        print(result)
    except Exception as e:
        print(f"Error during Ragas evaluation: {e}")

    print("Langchain-based RAGAS evaluation process completed.")

if __name__ == "__main__":
    main()

