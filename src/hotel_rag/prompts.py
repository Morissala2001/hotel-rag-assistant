"""The prompt sent to the language model."""

DEFAULT_HOTEL = "Casa Aurora Boutique Hotel"
REFUSAL = "I don't know, please contact the front desk."

INSTRUCTION = (
    "Answer in one or two sentences, using only the information in the documentation above. "
    f'If the information is not in the documentation, answer exactly: "{REFUSAL}"'
)


def role(hotel_name: str = DEFAULT_HOTEL) -> str:
    return f"You are the virtual assistant of {hotel_name}.\nHere is the hotel's official documentation:"


def build_prompt(context: str, question: str, hotel_name: str = DEFAULT_HOTEL) -> str:
    """Role, documentation, guest question, then the instruction that forbids making things up.

    The last sentence gives the model an honest way out: when the documentation does not say,
    it must answer with the fixed refusal instead of guessing.
    """
    return f"{role(hotel_name)}\n\n{context}\n\nGuest question: {question}\n{INSTRUCTION}"
