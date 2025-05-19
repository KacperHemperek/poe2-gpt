from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi_mcp import FastApiMCP
from google.genai import types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel
from utils import (
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


class ItemModel(BaseModel):
    name: str
    damage: str
    range: int
    weight: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.post("/ask", tags=["ai"], operation_id="ask_question", response_model=AskResponse)
async def ask_question(body: AskRequest):
    """
    Ask a question about POE items. The model can use available MCP tools when needed.
    """
    async with sse_client(
        "http://localhost:1337/mcp",
    ) as (read, write):
        async with ClientSession(read, write) as session:

            # TODO: Make it so the llm always returns relevant items in the answer and make sure
            # to put only a single item in the code block and not an array of elements
            # TODO: Save and use context of the chat
            # TODO: Make it possible to call multiple tools (not sure if it is possible with gemini and any other llm)

            # NOTE: might refactor this to make quering model recursive and make it possible to request multiple tool calls

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
