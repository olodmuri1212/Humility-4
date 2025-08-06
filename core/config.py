# core/config.py

import os
from dotenv import load_dotenv
import toml

# Load environment variables from a .env file at the project root
# This allows you to keep secret keys out of your code.
load_dotenv()

# Get API keys from environment
NON_REASONING_API_KEY = os.getenv("NON_REASONING_API_KEY")

# Example of how you might load other keys
# REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# --- API Keys & Configuration ---
# Fetch keys from the environment. They will be None if not set.
NON_REASONING_API_KEY = os.getenv("NON_REASONING_API_KEY")

if not NON_REASONING_API_KEY:
    print("Warning: NON_REASONING_API_KEY not found in .env file.")


# --- LLM Configuration for OpenRouter ---
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
# We use a state-of-the-art instruction-following model.
# You can swap this with other models available on OpenRouter.


# --- Prompt Loading ---
def load_prompts() -> dict:
    """Loads all prompts from the prompts.toml file for the agents."""
    try:
        # The path is relative to the project root where the app is run.
        with open("core/prompts.toml", "r", encoding="utf-8") as f:
            prompts = toml.load(f)
            print("Prompts loaded successfully from core/prompts.toml")
            return prompts
    except FileNotFoundError:
        print("FATAL ERROR: core/prompts.toml not found. The application cannot run without it.")
        return {}
    except Exception as e:
        print(f"FATAL ERROR: Could not read or parse core/prompts.toml: {e}")
        return {}

# Load prompts into a global constant for easy access across the application.
PROMPTS = load_prompts()


# --- Local Audio Processing Configuration ---
# Using local OpenAI Whisper for STT and pyttsx3 for TTS
# No API keys or external services required
