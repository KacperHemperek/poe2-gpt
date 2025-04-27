import json

from ai_client import client
from google.genai import types
from mcp import Tool as MCPTool
from mcp.types import CallToolResult


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


def query_model(
    content: types.ContentListUnion,
    function_declarations: list[types.FunctionDeclaration],
):
    """Query the gemini model allowing to use provided tools with the given tools"""

    return client.models.generate_content(
        model="gemini-2.0-flash",
        contents=content,
        config=types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=function_declarations)]
        ),
    )
