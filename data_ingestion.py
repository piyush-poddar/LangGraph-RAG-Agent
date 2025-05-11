import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from gemini_tools import GeminiEmbeddings

DOC_FOLDER = os.path.join(os.path.dirname(__file__), "documents")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

def load_all_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, filename), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename  # track which file it came from
            documents.extend(docs)
    return documents

def ingest_documents():
    print("🔄 Loading FAQ documents...")
    docs = load_all_documents(DOC_FOLDER)

    print(f"📄 Loaded {len(docs)} documents. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    chunk_contents = [chunk.page_content for chunk in chunks]

    print(f"🧠 Total chunks: {len(chunks)}. Generating embeddings...")
    #embeddings = []
    #for i in range(0,len(embeddings),100):
    #    embeddings.extend(embedding.values for embedding in get_embedding(chunk_contents[i:i+100]))
    
    embeddings = GeminiEmbeddings()

    print("💾 Storing in Chroma vector DB...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR)
    vectordb.persist()
    print("Ingestion complete!")


if __name__ == "__main__":
    if not os.path.exists(CHROMA_DIR):
        os.makedirs(CHROMA_DIR)
    ingest_documents()