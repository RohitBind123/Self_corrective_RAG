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