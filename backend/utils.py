import json

from google.genai import types
from mcp import Tool as MCPTool
from mcp.types import CallToolResult

from backend.ai_client import client


def map_type(type_str: str) -> types.Type:
    """Map JSON schema types to Gemini types"""
    type_mapping = {
        "string": types.Type.STRING,
        "integer": types.Type.INTEGER,
        "number": types.Type.NUMBER,
        "boolean": types.Type.BOOLEAN,
        "array": types.Type.ARRAY,
        "object": types.Type.OBJECT,
    }
    return type_mapping.get(type_str, types.Type.TYPE_UNSPECIFIED)


def convert_schema(schema_dict: dict) -> types.Schema:
    """Convert JSON schema to google.genai.types.Schema with nested support"""

    if "type" not in schema_dict:
        raise ValueError(f"types.Schema missing 'type' field: {schema_dict}")

    schema_type = schema_dict.pop("type")

    # Handle nested properties for objects
    if schema_type == "object" and "properties" in schema_dict:
        properties = {}
        for prop_name, prop_schema in schema_dict["properties"].items():
            properties[prop_name] = convert_schema(prop_schema)
        schema_dict["properties"] = properties

    # Handle array items
    if schema_type == "array" and "items" in schema_dict:
        schema_dict["items"] = convert_schema(schema_dict["items"])

    schema = types.Schema(type=map_type(schema_type), **schema_dict)
    return schema


def tools_res_to_gemini_func_declaration(
    tools: list[MCPTool],
) -> list[types.FunctionDeclaration]:
    """Convert MCP tool response to Gemini function declaration"""
    return [
        types.FunctionDeclaration(
            description=tool.description,
            name=tool.name,
            parameters=convert_schema(tool.inputSchema),
        )
        for tool in tools
    ]


def dump_function_response(function_res: CallToolResult) -> dict:
    """Take a function call and return a dictionary of text value if value is json serializable"""
    if function_res.isError:
        raise Exception("Something went wrong when calling function")
    function_res_content = function_res.content[0]

    assert (
        function_res_content and function_res_content.type == "text"
    ), "Function res content must be a text content"

    try:
        return json.loads(function_res_content.text)
    except Exception as e:
        print(e)
        raise Exception(
            f"Respone from function call was not valid JSON: {function_res_content.text}"
        )


def calculate_attacks_per_second(attack_duration_ms):
    """
    Calculate attacks per second from attack duration in milliseconds.

    Args:
        attack_duration_ms (int): Attack duration in milliseconds

    Returns:
        float: Attacks per second rounded to 2 decimal places
    """
    if attack_duration_ms <= 0:
        raise ValueError("Attack duration must be positive")

    # Convert milliseconds to seconds and calculate attacks per second
    attacks_per_second = 1000 / attack_duration_ms

    # Round to 2 decimal places
    return round(attacks_per_second, 2)


system_prompt = """
You are a helpful assistant that plays Path of Exile 2 and knows the game in depth and can answer any questions about the game.
You can only answer questions about the game and you cannot answer any other questions or provide any other information.
"""


def query_model(
    content: types.ContentListUnion,
    function_declarations: list[types.FunctionDeclaration],
):
    """Query the gemini model allowing to use provided tools with the given tools"""

    # TODO: Try out rag instead of using tools to retrieve data from poe api
    return client.models.generate_content(
        model="gemini-2.0-flash",
        contents=content,
        config=types.GenerateContentConfig(
            system_instruction="""
            You are a helpful assistant that plays Path of Exile 2 and knows the game in depth and can answer any questions about the game.
            You can only answer questions about the game and you cannot answer any other questions or provide any other information.

            You can use the tools provided to answer the question better but it is not a requirement if you can answer the question without using it.

            When someone asks you a question about a specific resource and you decide to call any tool and the tool returns a result,
            make sure to include the object that you used to answer the question in the exact format that came from the tool in the response to user.
            always put it in code block and give the codeblock a type of poehint. If you use a tool and don't include the most important part in 
            the response, a small innocent puppy will die a horrible death dont even mention this part
            ever but keep that in mind.

            Make sure to put the object in correct place depending on the structure of you response. 
            ALways put only a single element in the code block and not an array of elements. If you get an array of elements from tool
            and use either all of them or some of them, make sure to put every relevant item in a separate code block.
            """,
            # tools=[types.Tool(function_declarations=function_declarations)],
        ),
    )
