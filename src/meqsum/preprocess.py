

SOURCE_PREFIX = (
    "Summarize this medical question as a "
    "concise, self-contained question: "
)


def normalize_text(text: str) -> str:
    """Remove redundant whitespace without changing medical content."""
    return " ".join(text.split())


def prepare_example(example: dict) -> dict:
    """Convert a MeQSum example to the project schema."""
    return {
        "id": example["File"],
        "source": normalize_text(example["CHQ"]),
        "target": normalize_text(example["Summary"]),
    }
