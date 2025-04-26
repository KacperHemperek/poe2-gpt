from google.genai import types


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
