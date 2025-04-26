from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
from google import genai
from google.genai import types
from mcp import Tool as MCPTool
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel
from utils import convert_schema

load_dotenv()

app = FastAPI()
mcp = FastApiMCP(
    app,
    name="poe-knowledge",
    include_tags=["mcp"],
)
mcp.mount()

client = genai.Client()


class ItemModel(BaseModel):
    name: str
    damage: str
    range: int
    weight: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


def query_model(q: str, tools: list[MCPTool]):
    """Query the model with the given tools"""
    # TODO: Improve initial prompt to give answers even if no tools can be used
    # TODO: Save and use context of the chat
    # TODO:
    declarations: list[types.FunctionDeclaration] = [
        types.FunctionDeclaration(
            description=tool.description,
            name=tool.name,
            parameters=convert_schema(tool.inputSchema),
        )
        for tool in tools
    ]

    print("Declarations:")
    __import__("pprint").pprint(declarations)
    ai_res = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=q,
        config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=declarations)]
        ),
    )
    print("AI response:")
    __import__("pprint").pprint(ai_res)
    if ai_res.text:
        return AskResponse(answer=ai_res.text)


@app.post("/ask", tags=["ai"], operation_id="ask_question")
async def ask_question(body: AskRequest):
    """
    Ask a question about POE items. The model can use available MCP tools when needed.
    """
    async with sse_client(
        "http://localhost:1337/mcp",
    ) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            tools_res = await session.list_tools()
            declarations: list[types.FunctionDeclaration] = [
                types.FunctionDeclaration(
                    description=tool.description,
                    name=tool.name,
                    parameters=convert_schema(tool.inputSchema),
                )
                for tool in tools_res.tools
            ]

            ai_res = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=body.question,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(function_declarations=declarations)]
                ),
            )
            if ai_res.text:
                return AskResponse(answer=ai_res.text)


class GetItemsRequest(BaseModel):
    type: str


@app.get(
    "/mcp/get-items",
    response_model=list[ItemModel],
    tags=["items", "mcp"],
    operation_id="get_items",
)
async def get_items(req: GetItemsRequest) -> list[ItemModel]:
    """
    Get items of the given type and possible subtypes.

    Args:
        type (str): The type of the items to retrieve. Available types are "crossbow" and "bow".

    Returns:
        list[dict]: A list of items matching the given type.
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
    if not subitems:
        return []
    return subitems


@app.get("/health", tags=["health"], operation_id="health_check")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "ok"}


# NOTE: this needs to be called last after every route is setup to allow fastapi_mcp register all tools and their schemas etc
mcp.setup_server()
