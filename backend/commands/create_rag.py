from dotenv import load_dotenv
from langchain_core.documents import Document

from backend.commands.utils import (
    get_implicit_string,
    get_item_id,
    get_json_from_url,
    get_requirements_string,
    transform_to_metadata_dict,
)
from backend.db import chroma
from backend.utils import calculate_attacks_per_second


def get_mods_from_api() -> dict:
    """Retrieve all item mods from repoe api"""
    url = "https://repoe-fork.github.io/poe2/mods.min.json"
    res = get_json_from_url(url)
    # check type of res
    if not isinstance(res, dict):
        raise ValueError("Response is not a valid JSON object")
    return res


def get_item_strings(items: list, mods: dict) -> list[str]:
    """Transform items list to list of strings with explanations for the embedding model to work properly with it"""
    results = []
    for item in items:
        name = item.get("name")
        item_class = item.get("item_class")
        tags = ", ".join(item.get("tags"))
        requirements = item.get("requirements")
        drop_level = item.get("drop_level")
        implicits_ids = item.get("implicits")
        implicits = [mods[implicit_id] for implicit_id in implicits_ids]
        properties = item.get("properties")

        mapping = {
            "armour": "Armour",
            "energy_shield": "Energy Shield",
            "evasion": "Evasion",
            "movement_speed": "Movement Speed",
            "block": "Block",
            "description": "Description",
            "directions": "Directions",
            "stack_size": "Stack Size",
            "stack_size_currency_tab": "Stack Size Currency Tab",
            "full_stack_turns_into": "Full Stack Turns Into",
            "charges_max": "Charges Max",
            "charges_per_use": "Charges Per Use",
            "duration": "Duration",
            "life_per_use": "Life Per Use",
            "mana_per_use": "Mana Per Use",
            "attack_time": "Attack Time",
            "critical_strike_chance": "Critical Strike Chance",
            "physical_damage_max": "Physical Damage Max",
            "physical_damage_min": "Physical Damage Min",
            "range": "Range",
            "mana_burn_ms": "Mana Burn MS",
            "cooldown_ms": "Cooldown MS",
        }

        properties_strings = []
        for key, val in properties.items():
            if val is None or key not in mapping:
                print(f"Key {key} not in mapping or value is None: {val}")
                continue

            if key == "critical_strike_chance":
                val = f"{val / 100}%"
            elif key == "attack_time":
                # TODO: figure out what should be the unit for attact time
                atp = calculate_attacks_per_second(val)
                val = f"{atp} attacks per second"
                pass
            elif key == "movement_speed":
                # TODO: check if this is in percent or something else
                val = f"{val / 100}%"
            properties_strings.append(f"{mapping[key]}: {val}")

        string = f"""
        Name: {name}
        Type of an item, also known as item class: {item_class}
        Tags for item: {tags}
        {f"Requirents:\n{get_requirements_string(requirements)}" if requirements else ""}
        Minimum character level that item can drop on: {drop_level}
        Implicit item modifiers: {get_implicit_string(implicits)} 
        Properties of the item: {"\n".join(properties_strings)}
        """.replace(
            "\t", ""
        )
        results.append(string)
    return results


def main():
    """
    Command for creating the RAG database using chromadb and langchain, embedding data
    with Google API.
    """

    # Load all items from the repoe api and filter them to only include released items
    url = "https://repoe-fork.github.io/poe2/base_items.min.json"
    res = get_json_from_url(url)
    # check type of res
    if not isinstance(res, dict):
        raise ValueError("Response is not a valid JSON object")
    items = []
    for path, value in res.items():
        item_type = value.get("domain", None)
        release_state = value.get("release_state", None)
        if item_type == "item" and release_state == "released":
            value["path"] = path
            items.append(value)

    item_strings = get_item_strings(items, get_mods_from_api())

    # Insert all items into the database with their corresponding metadata
    docs: list[Document] = []
    for string, item in zip(item_strings, items):
        metadata = transform_to_metadata_dict(item)
        docs.append(Document(page_content=string, metadata=metadata))

    chroma.poe_store.add_documents(docs, ids=[get_item_id(item) for item in items])


if __name__ == "__main__":
    load_dotenv()
    main()
