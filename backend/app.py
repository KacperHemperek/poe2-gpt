from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain import chat_models
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from backend.db.chroma import poe_store
from backend.schemas import Message
from backend.utils import convert_to_langchain_message

load_dotenv()

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

llm = chat_models.init_chat_model(
    # model="gemini-2.5-flash-preview-04-17",
    model="gemini-2.0-flash",
    model_provider="google_genai",
)


@tool(
    response_format="content_and_artifact",
    # description="Retrieve all available data on any information about ingame Path of Exile 2 things. Including items, skills, modificators and anything else.",
)
def retrieve(query: str):
    """Retrieve the context for the question from rag"""
    context = poe_store.similarity_search(query, k=15)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\n Content: {doc.page_content}" for doc in context
    )
    return serialized, context


def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Step 2: Execute the retrieval.
tools = ToolNode([retrieve])


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know."
        "Provide explanations and reasoning for choosen answer. "
        "If the question is related to some game mechanics, try to explan it in "
        "a way that a new player can understand it unless specified otherwise. "
        "\n\n"
        f"{docs_content}"
    )

    conversation_messages = []
    for message in state["messages"]:
        if message.type in ("human", "system") or (
            message.type == "ai" and not message.tool_calls
        ):
            conversation_messages.append(message)
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = llm.invoke(prompt)
    return {"messages": [response]}


graph_builder = StateGraph(MessagesState)
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


class AskRequest(BaseModel):
    messages: List[Message]


class AskResponse(BaseModel):
    messages: List[Message]


@app.post("/ask", tags=["ai"], operation_id="ask")
async def ask_question(body: AskRequest):
    """
    Ask a question about POE items. The model will use the RAG database to retrieve the context.
    """
    messages = [convert_to_langchain_message(m) for m in body.messages]
    result = graph.invoke({"messages": messages})
    return AskResponse(messages=result["messages"])
