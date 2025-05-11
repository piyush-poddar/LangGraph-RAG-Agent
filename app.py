import streamlit as st
from agent import start_workflow 

# Initialize the graph only once
@st.cache_resource
def init_graph():
    return start_workflow()

graph = init_graph()

# App title
#st.set_page_config(page_title="Multi-Tool Assistant", layout="centered")
st.title("🧠 Multi-Tool RAG Assistant")
st.markdown("Ask anything! This assistant uses **RAG**, a **Calculator**, and a **Dictionary** tool intelligently based on your query.")

# Session state to persist messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box for user query
user_input = st.chat_input("Ask me something...")
if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Append to session
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call the graph
    result = graph.invoke({"messages": st.session_state.messages}, config={"configurable": {"thread_id": "1"}})

    # Extract latest tool output
    if "messages" in result:
        new_message = result["messages"][-1]
        with st.chat_message("assistant"):
            if new_message.get("tool_used") == "rag":
                st.markdown(f"**Tool Used:** {new_message.get('tool_used', 'RAG').capitalize()}\n\n**Retrieved Context Snippets:** \n\n{new_message['context']}\n\n**Result:** {new_message['content']}")
            else:
                st.markdown(f"**Tool Used:** {new_message.get('tool_used', 'RAG').capitalize()}\n\n**Result:** {new_message['content']}")
        if new_message.get("tool_used") == "rag":
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"**Tool Used:** {new_message.get('tool_used', 'RAG').capitalize()}\n\n**Retrieved Context Snippets:** \n\n{new_message['context']}\n\n**Result:** {new_message['content']}"
            })
        else:
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"**Tool Used:** {new_message.get('tool_used', 'RAG').capitalize()}\n\n**Result:** {new_message['content']}"
            })
