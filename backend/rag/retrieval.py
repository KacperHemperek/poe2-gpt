import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
chroma_client = chromadb.HttpClient()
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

poe_store = Chroma(
    collection_name="poe",
    embedding_function=embeddings,
    client=chroma_client,
)
res = poe_store.similarity_search(
    "What is the best bow in for lightning arrow build?", k=6
)

__import__("pprint").pprint(res)
