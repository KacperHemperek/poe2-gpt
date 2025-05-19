import chromadb
from langchain_chroma import Chroma

from backend.rag.embeddings import embeddings

chroma_client = chromadb.HttpClient()

poe_store = Chroma(
    collection_name="poe",
    embedding_function=embeddings,
    client=chroma_client,
)
