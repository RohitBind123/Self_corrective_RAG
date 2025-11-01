import os
import dill
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import OllamaEmbeddings
from src.crag.config import URLS, EMBEDDING_MODEL, VECTOR_STORE_PATH

def create_vector_store():
    """Create the vector store from the URLs and save it to a file."""
    print("Loading documents from URLs...")
    docs = [WebBaseLoader(url).load() for url in URLS]
    docs_list = [item for sublist in docs for item in sublist]

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    print(f"Creating embeddings with {EMBEDDING_MODEL}...")
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print("Creating vector store...")
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embedding,
    )

    print(f"Saving vector store to {VECTOR_STORE_PATH}...")
    with open(VECTOR_STORE_PATH, "wb") as f:
        dill.dump(vectorstore, f)

    print("Vector store created successfully.")

def load_vector_store():
    """Load the vector store from the file."""
    if not os.path.exists(VECTOR_STORE_PATH):
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_STORE_PATH}. "
            f"Please run the indexing script first."
        )

    with open(VECTOR_STORE_PATH, "rb") as f:
        return dill.load(f)
