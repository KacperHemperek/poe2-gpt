import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.commands.utils import get_item_id, transform_to_metadata_dict
from backend.rag.embeddings import embeddings

chroma_client = chromadb.HttpClient()

poe_store = Chroma(
    collection_name="poe",
    embedding_function=embeddings,
    client=chroma_client,
)


def insert_items(item_strings: list[str], items: list[dict]):
    """Insert items into the database."""
    docs: list[Document] = []
    for string, item in zip(item_strings, items):
        metadata = transform_to_metadata_dict(item)
        docs.append(Document(page_content=string, metadata=metadata))

    poe_store.add_documents(docs, ids=[get_item_id(item) for item in items])


def insert_unique_items():
    """Insert unique items into the database."""
    pass


def insert_mods():
    """Insert mods into the database."""
    pass


def insert_skills():
    """Insert skills into the database."""
    pass


def insert_support_gems():
    """Insert support gems into the database."""
