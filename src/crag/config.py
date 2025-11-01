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
