# Corrective RAG (CRAG) Project

This project implements a Corrective RAG (CRAG) pipeline using local LLMs with LangGraph.

## Setup

1.  **Install dependencies:**

    ```bash
    uv sync
    ```

2.  **Set up environment variables:**

    Create a `.env` file in the root of the project and add your Tavily API key:

    ```
    TAVILY_API_KEY=your_tavily_api_key
    ```

3.  **Create the vector store:**

    ```bash
    .venv/bin/python -m scripts.create_index
    ```

## Running the application

To run the FastAPI application, use the following command:

```bash
.venv/bin/python main.py
```

The application will be available at `http://localhost:8000`.

## API

You can send a POST request to the `/invoke` endpoint with a JSON payload like this:

```json
{
  "input": "What are the types of agent memory?"
}
```

You can use `curl` to test the endpoint:

```bash
curl -X POST "http://localhost:8000/invoke" -H "Content-Type: application/json" -d '{"input": "What are the types of agent memory?"}'
```
