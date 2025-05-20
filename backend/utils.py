from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from backend.schemas import Message


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


def convert_to_langchain_message(message: Message):
    """Convert a message to a langchain message"""
    if message.role == "human":
        return HumanMessage(content=message.content)
    elif message.role == "ai":
        return AIMessage(content=message.content)
    elif message.role == "system":
        return SystemMessage(content=message.content)
    elif message.role == "tool":
        return ToolMessage(
            content=message.content,
        )
    else:
        raise ValueError(
            f"Unknown message role: {message.role}, content: {message.content}"
        )


def convert_to_client_message(message: BaseMessage):
    """Convert a langchain message to a client message"""
    if not isinstance(message.content, str):
        raise ValueError(f"Message content is not a string: {message.content}")
    if message.type == "human":
        return Message(role="human", content=message.content)
    elif message.type == "ai":
        return Message(role="ai", content=message.content)
    elif message.type == "system":
        return Message(role="system", content=message.content)
    else:
        raise ValueError(
            f"Unknown message type: {message.type}, content: {message.content}"
        )
