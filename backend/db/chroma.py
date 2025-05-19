import chromadb
from langchain_chroma import Chroma
from rag.embeddings import embeddings

chroma_client = chromadb.HttpClient()

item_store = Chroma(
    collection_name="items",
    embedding_function=embeddings,
    client=chroma_client,
)
