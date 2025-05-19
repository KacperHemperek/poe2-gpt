from db import chroma
from dotenv import load_dotenv

load_dotenv()


response = chroma.item_store.similarity_search(
    "give me all bows that have increased chain chance", k=5
)

__import__("pprint").pprint(response)
