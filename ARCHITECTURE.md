# Self-Corrective RAG Architecture

This document outlines the architecture of the self-corrective Retrieval-Augmented Generation (RAG) model.

## High-Level Architecture

A self-corrective RAG is a system that improves its own knowledge base over time by identifying and correcting errors in its retrieved information. It typically consists of a few core components:

1.  **Retriever:** Retrieves relevant documents from a knowledge base (e.g., a vector store) based on a user query.
2.  **Generator:** Generates an answer to the user query based on the retrieved documents.
3.  **Critic:** Evaluates the generated answer and the retrieved documents for accuracy, relevance, and overall quality.
4.  **Corrector:** If the critic identifies errors, the corrector updates the knowledge base or the retrieval strategy to prevent similar errors in the future.

This cyclical process of retrieval, generation, critique, and correction allows the model to continuously learn and improve from its mistakes, leading to more accurate and reliable answers over time.

## Low-Level Architecture (Brief)

*   **Retriever:** This can be implemented using dense vector representations of the documents in the knowledge base. A model like `e5-large-v2` can be used to generate these embeddings, and a vector store like FAISS or Chroma can be used for efficient similarity search.

*   **Generator:** A powerful large language model (LLM) is used to generate the final answer. This could be a model from the GPT family, Llama, or a fine-tuned open-source model.

*   **Critic:** The critic can be another LLM, prompted to evaluate the generated answer against the source documents. It can also be a set of smaller, more specialized models trained to detect specific types of errors, such as factual inconsistencies, irrelevance, or hallucinations. Human-in-the-loop feedback can also be integrated at this stage.

*   **Corrector:** This is the most complex component. The correction mechanism can be implemented in several ways:
    *   **Document Editing:** Directly modifying the content of the source documents if they are found to be incorrect or outdated.
    *   **Re-indexing:** Updating the vector representations of the documents in the knowledge base to reflect the corrections.
    *   **Retriever Fine-Tuning:** Adjusting the retriever model itself to improve its ability to rank relevant documents for a given query.
    *   **Negative Feedback:** Storing information about incorrect retrievals or generated answers to avoid similar mistakes in the future. This can be done by adding negative examples to the training data for the retriever or generator.

## Detailed Architecture and Code Explanation

### 1. Project Entrypoint

The application is served using a FastAPI server. The entrypoint is `main.py`, which starts the `uvicorn` server.

```python
# main.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.crag.app:app", host="0.0.0.0", port=8000, reload=True)
```

The FastAPI application is defined in `src/crag/app.py`. It exposes a `/crag` endpoint that accepts a question and returns the generated answer.

```python
# src/crag/app.py
from fastapi import FastAPI
from langserve import add_routes
from src.crag.graph import custom_graph
from pydantic import BaseModel
from langchain_core.runnables import RunnableLambda

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

# 1. Define the input model
class Input(BaseModel):
    question: str

# 2. Define a function to transform the input to the graph's expected state
def get_initial_state(input: dict):
    return {"question": input['question'], "steps": []}

# 3. Create the final chain
chain = RunnableLambda(get_initial_state) | custom_graph

# 4. Add the routes
add_routes(
    app,
    chain.with_types(input_type=Input),
    path="/crag",
)
```

### 2. The Core Logic: `langgraph` State Machine

The heart of the application is the `StateGraph` defined in `src/crag/graph.py`. This graph represents the workflow of the RAG pipeline.

#### Graph State

The state of the graph is defined by the `GraphState` class, which is a `TypedDict`. It holds the data that is passed between the nodes of the graph.

```python
# src/crag/graph.py
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        search: whether to add search
        documents: list of documents
        steps: List[str]
    """

    question: str
    generation: str
    search: str
    documents: List[Document]
    steps: List[str]
```

#### Graph Nodes

The graph consists of several nodes, each performing a specific task:

*   **`retrieve`**: This is the first node. It takes the user's question and retrieves relevant documents from the vector store.

    ```python
    # src/crag/graph.py
    def retrieve(state):
        """
        Retrieve documents
        """
        question = state["question"]
        documents = retriever.invoke(question)
        steps = state.get("steps", []) + ["retrieve_documents"]
        return {"documents": documents, "question": question, "steps": steps}
    ```

*   **`grade_documents`**: This node assesses the relevance of each retrieved document. It uses an LLM-based grader (`retrieval_grader`) to assign a "yes" or "no" score. If any document is graded as "no", it sets the `search` flag to "Yes".

    ```python
    # src/crag/graph.py
    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.
        """
        question = state["question"]
        documents = state["documents"]
        steps = state.get("steps", []) + ["grade_document_retrieval"]
        filtered_docs = []
        search = "No"
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "documents": d.page_content}
            )
            grade = score["score"]
            if grade == "yes":
                filtered_docs.append(d)
            else:
                search = "Yes"
                continue
        return {
            "documents": filtered_docs,
            "question": question,
            "search": search,
            "steps": steps,
        }
    ```

*   **`web_search`**: If the `grade_documents` node decides that a web search is needed, this node is executed. It uses the Tavily API to search the web for the user's question and adds the search results to the list of documents.

    ```python
    # src/crag/graph.py
    def web_search(state):
        """
        Web search based on the re-phrased question.
        """
        question = state["question"]
        documents = state.get("documents", [])
        steps = state.get("steps", []) + ["web_search"]
        web_results = web_search_tool.invoke({"query": question})
        documents.extend(
            [
                Document(page_content=d["content"], metadata={"url": d["url"]})
                for d in web_results
            ]
        )
        return {"documents": documents, "question": question, "steps": steps}
    ```

*   **`generate`**: This is the final step in the process. It takes the final set of documents (either from the initial retrieval or from the web search) and generates a concise answer to the user's question.

    ```python
    # src/crag/graph.py
    def generate(state):
        """
        Generate answer
        """
        question = state["question"]
        documents = state["documents"]
        generation = rag_chain.invoke({"documents": documents, "question": question})
        steps = state.get("steps", []) + ["generate_answer"]
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "steps": steps,
        }
    ```

#### Graph Construction

The nodes are connected together to form the graph. The `add_conditional_edges` function is used to implement the self-correction logic. Based on the output of the `grade_documents` node, the graph will either proceed to `generate` or go to `web_search`.

```python
# src/crag/graph.py
# Graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("web_search", web_search)

# Build graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "search": "web_search",
        "generate": "generate",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

custom_graph = workflow.compile()
```

### 3. Indexing and Vector Store

The `src/crag/services/indexing.py` module is responsible for creating and loading the vector store.

*   **`create_vector_store`**: This function loads documents from a list of URLs, splits them into smaller chunks, generates embeddings using the `OllamaEmbeddings` model, and stores them in a `SKLearnVectorStore`. The vector store is then saved to a file using `dill`.

    ```python
    # src/crag/services/indexing.py
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
    ```

*   **`load_vector_store`**: This function loads the pre-built vector store from the file.

    ```python
    # src/crag/services/indexing.py
    def load_vector_store():
        """Load the vector store from the file."""
        if not os.path.exists(VECTOR_STORE_PATH):
            raise FileNotFoundError(
                f"Vector store not found at {VECTOR_STORE_PATH}. "
                f"Please run the indexing script first."
            )

        with open(VECTOR_STORE_PATH, "rb") as f:
            return dill.load(f)
    ```

### 4. Configuration

The `src/crag/config.py` file holds all the important configurations for the project, such as API keys, model names, and the path to the vector store.

```python
# src/crag/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Models
LLM_MODEL = "llama3:8b"
EMBEDDING_MODEL = "nomic-embed-text:v1.5"

# Vector Store
VECTOR_STORE_PATH = "vectorstore.pkl"

# Data
URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
```

## Conclusion

This project provides a practical implementation of a self-corrective RAG pipeline. By using a graph-based approach with `langgraph`, it creates a flexible and extensible system that can be easily modified and improved. The self-correction mechanism, which involves grading the retrieved documents and performing a web search when necessary, is a key feature that enhances the accuracy and reliability of the generated answers.