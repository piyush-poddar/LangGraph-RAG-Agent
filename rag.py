import os
import google.generativeai as gemini
from gemini_tools import GeminiEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "chroma_db"

gemini.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
model = gemini.GenerativeModel("gemini-1.5-flash")

def query_vector_db(query: str, k: int = 3):
    print(f"🔍 Searching for: {query}")
    
    # Load existing vectorstore
    embedding = GeminiEmbeddings()
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding
    )

    # Embed query and retrieve top-k results
    results = vectordb.similarity_search(query, k=k)
    
    return [doc.page_content for doc in results]
    # print(f"Top {k} results:\n")
    # for i, doc in enumerate(results, 1):
    #     print(f"{i}. {doc.page_content.strip()}")
    #     print(f"   Source: {doc.metadata.get('source')}\n")

def get_rag_response(query: str, context: list) -> str:
    """
    Sends a query and its context to Gemini 1.5 Flash and returns the LLM-generated response.
    
    Parameters:
        query (str): The original user question.
        context (str or list of str): Retrieved documents or chunks from the vector DB.

    Returns:
        str: Gemini-generated answer.
    """
    # if isinstance(context, list):
    context = "\n\n".join(context)

    prompt = f"""You are a helpful assistant. Use the provided context to answer the question accurately.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating response from Gemini: {e}"

if __name__ == "__main__":
    query = "does blinkit deliver ciggarettes?"
    #k = int(input("Enter number of results to retrieve (default 3): ") or 3)
    print(get_rag_response(query, query_vector_db(query)))