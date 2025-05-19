from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi_mcp import FastApiMCP
from google.genai import types
from langchain import chat_models, hub
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel

from backend.db.chroma import poe_store
from backend.utils import (
    dump_function_response,
    query_model,
    tools_res_to_gemini_func_declaration,
)

load_dotenv()

app = FastAPI()
mcp = FastApiMCP(
    app,
    name="poe-knowledge",
    include_tags=["mcp"],
)
mcp.mount()

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
    __import__("pprint").pprint(context)
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


class ItemModel(BaseModel):
    name: str
    damage: str
    range: int
    weight: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.post("/ask", tags=["ai"], operation_id="ask")
async def ask_question(body: AskRequest):
    """
    Ask a question about POE items. The model will use the RAG database to retrieve the context.
    """
    q = body.question
    user_message = HumanMessage(content=q)
    result = graph.invoke({"messages": [user_message]})
    print("=============RESPONSE===================")
    __import__("pprint").pprint(result["messages"][-1])
    return AskResponse(answer=result["messages"][-1].content)


@app.post(
    "/ask-with-tools",
    tags=["ai"],
    operation_id="ask_question",
    response_model=AskResponse,
)
async def ask_question_with_tools(body: AskRequest):
    """
    Ask a question about POE items. The model can use available MCP tools when needed.
    """
    async with sse_client(
        "http://localhost:1337/mcp",
    ) as (read, write):
        async with ClientSession(read, write) as session:

            # Initialize the session and retrieve the list of tools
            await session.initialize()
            tools_res = await session.list_tools()
            declarations = tools_res_to_gemini_func_declaration(tools_res.tools)
            content: types.ContentListUnion = [
                types.UserContent(parts=[types.Part.from_text(text=body.question)]),
            ]

            # Ask the question to the model with the tools included
            ai_res = query_model(content, declarations)

            # If ai responded with text return it immediately
            if ai_res.text:
                return AskResponse(answer=ai_res.text)

            # Make sure that the AI response has all the required elements
            assert ai_res.candidates, "AI Response must have candidates"
            assert ai_res.candidates[0].content, "First AI Response must have content"
            assert (
                ai_res.candidates[0].content.parts
                and len(ai_res.candidates[0].content.parts) == 1
            ), "Content must have exactly one part"

            part = ai_res.candidates[0].content.parts[0]

            # Call functions if the AI requested and reprompt the model with result from the tool call
            if part.function_call and part.function_call.name:
                assert (
                    part.function_call.name and part.function_call.args
                ), "Function call must have name and args"

                function_res = await session.call_tool(
                    part.function_call.name, part.function_call.args
                )

                dict_function_res = dump_function_response(function_res)

                content.append(
                    types.ModelContent(
                        parts=[
                            types.Part.from_function_call(
                                name=part.function_call.name,
                                args=part.function_call.args,
                            )
                        ]
                    ),
                )
                content.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=part.function_call.name,
                                response=dict_function_res,
                            )
                        ],
                    ),
                )

                ai_res = query_model(content, declarations)
                if ai_res.text:
                    return AskResponse(answer=ai_res.text)
            raise HTTPException(
                status_code=500,
                detail="Ai response did not have text nor function calls",
            )


class GetItemsRequest(BaseModel):
    type: str


class GetItemsResponse(BaseModel):
    items: list[ItemModel]


@app.post(
    "/mcp/get-items",
    response_model=GetItemsResponse,
    tags=["items", "mcp"],
    operation_id="get_items",
)
async def get_items(req: GetItemsRequest) -> GetItemsResponse:
    """
    Get items of the given type and possible subtypes.

    Args:
        type (str): The type of the items to retrieve. Available types are "crossbow" and "bow".

    Returns:
        GetItemsResponse: A response containing items field with a list of all items of the given type
    """
    type = req.type.lower()

    # TODO: make it request poe2 api with all items to retrieve bases
    items: dict[str, list[ItemModel]] = {
        "crossbow": [
            ItemModel(
                name="Heavy Crossbow",
                damage="8-10",
                range=120,
                weight=5,
            ),
            ItemModel(
                name="Light Crossbow",
                damage="6-8",
                range=80,
                weight=3,
            ),
        ],
        "bow": [
            ItemModel(
                name="Longbow",
                damage="6-8",
                range=150,
                weight=3,
            ),
            ItemModel(
                name="Shortbow",
                damage="4-6",
                range=80,
                weight=2,
            ),
        ],
    }
    subitems = items.get(type, [])
    return GetItemsResponse(items=subitems)


@app.get("/health", tags=["health"], operation_id="health_check")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "ok"}


# NOTE: this needs to be called last after every route is setup to allow fastapi_mcp register all tools and their schemas etc
mcp.setup_server()
