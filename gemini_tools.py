import os
from google import genai
from langchain_core.embeddings import Embeddings

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))

class GeminiEmbeddings(Embeddings):
    def __init__(self):
        self.client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self.client.models.embed_content(
                model="text-embedding-004",
                contents=texts
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            print(f"Error embedding documents: {e}")
            return []

    def embed_query(self, text: str) -> list[float]:
        try:
            response = self.client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            print(f"Error embedding query: {e}")
            return []

