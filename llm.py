import os
from groq import Groq
from typing import Optional
import logging
from dotenv import load_dotenv

logger = logging.getLogger("LLM-Wrapper")

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    logger.error(" GROQ_API_KEY is missing. Please set it in your environment.")

client = Groq(api_key=API_KEY)


DEFAULT_MODEL = "llama-3.3-70b-versatile"

def generate_llm_response(
    system_prompt: str,
    focus_prompt: str,
    temperature: float = 0.2,
    model: str = DEFAULT_MODEL,
    max_tokens: Optional[int] = None,
    json_mode: bool = False
) -> str:
    """
    Wrapper for Groq API that matches the interface expected by your nodes.
    """
    try:

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": focus_prompt}
        ]
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        if json_mode:
            params["response_format"] = {"type": "json_object"}
            if "json" not in system_prompt.lower():
                 messages[0]["content"] += " You must respond in JSON."

        completion = client.chat.completions.create(**params)

        return completion.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"Groq API Error: {e}")
        return f"Error connecting to AI: {str(e)}"