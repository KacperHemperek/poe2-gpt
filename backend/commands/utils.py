import chromadb
import requests


def get_json_from_url(url, params=None, headers=None, timeout=30):
    """
    Retrieve JSON data from a URL using HTTP GET request.

    Args:
        url (str): The URL to send the request to
        params (dict, optional): Dictionary of URL parameters to append to the URL
        headers (dict, optional): Dictionary of HTTP headers
        timeout (int, optional): Request timeout in seconds

    Returns:
        dict: The JSON response data

    Raises:
        requests.exceptions.RequestException: If the request fails
        ValueError: If the response is not valid JSON
    """
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        raise e
    except ValueError as e:
        raise ValueError(f"Invalid JSON response: {e}")


def transform_to_metadata_dict(item: dict) -> chromadb.Metadata:
    tags = (
        ", ".join(item["tags"])
        if item["tags"] and isinstance(item["tags"], list)
        else ""
    )
    implicits = (
        ", ".join(item["implicits"])
        if item["implicits"] and isinstance(item["implicits"], list)
        else ""
    )

    return {
        "name": item["name"],
        "item_class": item["item_class"],
        "tags": tags,
        # "requirements": requirements,
        "drop_level": item["drop_level"],
        "implicits": implicits,
        "path": item["path"],
    }


def get_item_id(item: dict) -> str:
    path = item["path"]
    if not path or not isinstance(path, str):
        raise ValueError(f"Invalid path: {path}")
    return path


def get_requirements_string(requirements: dict) -> str:
    result = ""
    for key, val in requirements.items():
        if val != 0:
            result += f"{key}: {val}\n"
    return result


def get_implicit_string(implicits: list) -> str:
    implicit_strings = []
    for implicit in implicits:
        description = implicit.get("text")
        stats = implicit.get("stats")
        stat_strings = []
        for stat in stats:
            max_value = stat.get("max")
            min_value = stat.get("min")
            stat_strings.append(f"Max and min range for mod: {min_value} - {max_value}")
        stat_string = "\n".join(stat_strings)
        implicit_strings.append(
            f"""
        Implicit description: {description}
        Implicit stats: {stat_string}
        """
        )

    return "\n".join(implicit_strings)
