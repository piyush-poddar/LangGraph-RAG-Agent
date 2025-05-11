import os
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import google.generativeai as genai
from rag import query_vector_db, get_rag_response
from typing import Annotated, TypedDict

# Configure Google API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# Define the calculator tool
@tool
def calculator_tool(input: str) -> dict:
    """
    Evaluate a simple mathematical expression provided as a string.
    """
    try:
        result = eval(input, {"__builtins__": {}})
        return {"tool_used": "calculator", "result": str(result)}
    except Exception:
        return {"tool_used": "calculator", "result": "Invalid calculation expression."}

# Define the dictionary tool
@tool
def dictionary_tool(word: str) -> dict:
    """
    Look up the definition of a word using the Dictionary API.
    """
    try:
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}")
        if response.status_code == 200:
            data = response.json()
            definition = data[0]["meanings"][0]["definitions"][0]["definition"]
            return {"tool_used": "dictionary", "result": definition}
        else:
            return {"tool_used": "dictionary", "result": "Definition not found for the given word."}
    except Exception as e:
        return {"tool_used": "dictionary", "result": f"Error accessing dictionary: {e}"}

# Define the RAG tool
@tool
def rag_tool(query: str) -> dict:
    """
    A tool that provides RAG, retrieval-augmented generation.
    """
    context = query_vector_db(query)
    llm_answer = get_rag_response(query, context)
    return {"tool_used": "rag", "context":"".join(f"{i+1}. {context[i]}\n\n" for i in range(len(context))), "result": llm_answer}

# Define the State type
class State(TypedDict):
    messages: list

# Define the calculator node
def calculator_node(state: State) -> State:
    query = state["messages"][-1]["content"]
    tool_response = calculator_tool.invoke(query.split()[-1]) 
    return {"messages": state["messages"] + [{"role": "tool", "content": tool_response["result"], "tool_used": tool_response["tool_used"]}]}

# Define the define node
def define_node(state: State) -> State:
    query = state["messages"][-1]["content"]
    tool_response = dictionary_tool.invoke(query.split()[-1])
    return {"messages": state["messages"] + [{"role": "tool", "content": tool_response["result"], "tool_used": tool_response["tool_used"]}]}

# Define the RAG node
def rag_node(state: State) -> State:
    query = state["messages"][-1]["content"]
    tool_response = rag_tool.invoke(query)
    return {"messages": state["messages"] + [{"role": "tool", "content": tool_response["result"], "tool_used": tool_response["tool_used"], "context": tool_response["context"]}]}

# Define the tool router function
def tool_router(state):
    user_query = state["messages"][-1]["content"].lower()

    if any(keyword in user_query for keyword in ["calculate", "add", "subtract", "multiply", "divide"]):
        return "calculator"
    elif any(keyword in user_query for keyword in ["define", "meaning of", "what is the definition of"]):
        return "define"
    else:
        return "rag"

# Start the workflow by building the graph
def start_workflow():
    graph_builder = StateGraph(State)
    graph_builder.add_node("calculator", calculator_node)
    graph_builder.add_node("define", define_node)
    graph_builder.add_node("rag", rag_node)

    graph_builder.set_entry_point("router")
    graph_builder.add_node("router", lambda state: state)

    graph_builder.add_conditional_edges("router", tool_router, {
        "calculator": "calculator",
        "define": "define",
        "rag": "rag"
    })

    # Finalize
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    #graph = graph_builder.compile(checkpointer=memory)
    # try:
    #     display(Image(graph.get_graph().draw_mermaid_png()))
    # except Exception:
    #     print("Error generating graph visualization.")
    return graph

# Main execution for testing
if __name__ == "__main__":
    graph = start_workflow()
    config = {"configurable": {"thread_id": "1"}}
    r = graph.invoke({"messages": [{"role": "user", "content": "calculate 5+2"}]}, config)
    print(r)
