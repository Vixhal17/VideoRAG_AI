import os
from langchain_mistralai import ChatMistralAI

def get_llm(temperature: float = 0.3) -> ChatMistralAI:
    """Centralized factory to initialize and return a ChatMistralAI client.
    
    Args:
        temperature (float): Controls response randomness. Defaults to 0.3.
    """
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=temperature
    )
