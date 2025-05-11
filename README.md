# 🔮 LangGraph-RAG-Agent

**LangGraph-RAG-Agent** is a multi-tool, LangGraph-powered intelligent assistant built using **Google Gemini 1.5 Flash**. It integrates **Retrieval-Augmented Generation (RAG)**, a **calculator**, and a **dictionary** with conditional routing and memory support — all packed into a simple **Streamlit** interface.

---

## 🧠 Architecture Overview

### Core Components:
- **LLM Backend**: Google Gemini 1.5 Flash (`ChatGoogleGenerativeAI`)
- **Workflow Engine**: LangGraph (for graph-based agent orchestration)

### Tools Integrated:
- ✅ **Calculator**: Handles math expressions.
- 📖 **Dictionary**: Fetches definitions via `dictionaryapi.dev`.
- 🔍 **RAG**: Custom vector search using `query_vector_db`.

### Additional Features:
- **Tool Routing**: Keyword-based tool selector (router node).
- **Memory**: LangGraph `MemorySaver` for checkpointing state.
- **Frontend**: Streamlit app for user interaction.

---

## ⚙️ Key Design Choices

- **LangGraph + ToolNode**: Modular support for tools and flow logic.
- **TypedDict State**: Tracks message history.
- **@tool Decorators**: Simplifies tool definition and invocation.
- **Streamlit**: Enables quick prototyping of LLM + tool workflows.
- **Google Gemini API**: Provides fast and context-aware generation.
- **Keyword-Based Routing**: Lightweight and efficient tool selection.

---

## 🚀 How to Run

### 1. 📦 Install Requirements
Run the following command to install dependencies:
```bash
pip install -r requirements.txt
```

### 2. 🔑 Set Up Your API Key
Set your Google API key in your environment:
```bash
export GOOGLE_API_KEY="your-api-key"
```

### 3. ▶️ Launch the App
```bash
streamlit run app.py
```

---

## 💬 Example Queries
```text
calculate 5+7
define serendipity
Does blinkit deliver ciggarettes?
```
