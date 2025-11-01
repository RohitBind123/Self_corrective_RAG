from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_ollama import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults

from src.crag.config import LLM_MODEL
from src.crag.services.indexing import load_vector_store

# Load the vector store and create a retriever
vectorstore = load_vector_store()
retriever = vectorstore.as_retriever(k=4)

# Define the LLM
llm = ChatOllama(model=LLM_MODEL, temperature=0)

# Retrieval Grader
retrieval_prompt = PromptTemplate(
    template="""You are a grader assessing the relevance of a retrieved document to a user question.
    Here is the retrieved document:
    {documents}

    Here is the user question:
    {question}

    If the document contains keywords related to the user question, grade it as relevant.
    Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
    Your output MUST be only the JSON object.
    """,
    input_variables=["question", "documents"],
)

retrieval_grader = retrieval_prompt | llm | JsonOutputParser()

# Generate
generation_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.

    Use the following documents to answer the question.

    If you don't know the answer, just say that you don't know.

    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)

rag_chain = generation_prompt | llm | StrOutputParser()

# Web Search Tool
web_search_tool = TavilySearchResults(k=3)


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


def retrieve(state):
    """
    Retrieve documents
    """
    question = state["question"]
    documents = retriever.invoke(question)
    steps = state.get("steps", []) + ["retrieve_documents"]
    return {"documents": documents, "question": question, "steps": steps}


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


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.
    """
    search = state["search"]
    if search == "Yes":
        return "search"
    else:
        return "generate"


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
