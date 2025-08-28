# # core/agents.py

# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.exceptions import OutputParserException

# from .config import NON_REASONING_API_KEY, OPENROUTER_API_BASE, PROMPTS
# from .state import AgentScore

# # --- 1. Reusable LLM and Parser Setup ---

# # Initialize the LLM client to communicate with OpenRouter's API.
# # We configure it once here and reuse it for all agents.
# llm = ChatOpenAI(
#     model="google/gemini-2.0-flash-001",
#     openai_api_key=NON_REASONING_API_KEY,
#     openai_api_base=OPENROUTER_API_BASE,
#     temperature=0.1,  # Low temperature for consistent, analytical output
#     max_tokens=150,
#     model_kwargs={
#         # This header is recommended by OpenRouter for tracking.
#         "extra_headers": {"HTTP-Referer": "http://localhost:8501"}
#     },
# )

# # Initialize a JSON parser that will try to shape the LLM's output
# # into our Pydantic 'AgentScore' model.
# json_parser = JsonOutputParser(pydantic_object=AgentScore)


# # --- 2. Agent Factory (for LLM-based agents) ---

# def create_analysis_agent(agent_name: str):
#     """
#     A factory function to dynamically create a LangChain agent (a "chain")
#     for any given humility trait defined in prompts.toml.

#     Args:
#         agent_name: The name of the agent, which must match a section in prompts.toml.

#     Returns:
#         A LangChain runnable sequence (a "chain") ready to be invoked.
#     """
#     if agent_name not in PROMPTS:
#         raise ValueError(f"Prompt for agent '{agent_name}' not found in prompts.toml")

#     # Combine the agent's specific system prompt with the general JSON formatting instructions.
#     # This is a crucial step for reliable JSON output.
#     prompt_text = (
#         PROMPTS[agent_name]["system_prompt"]
#         + "\n\n"
#         + PROMPTS["General"]["json_format_instructions"]
#         + "\n\nCandidate's response: {transcript}"
#     )

#     prompt_template = ChatPromptTemplate.from_template(prompt_text)

#     # Create the chain by piping the components together:
#     # 1. The input dictionary goes into the prompt template.
#     # 2. The formatted prompt goes to the LLM.
#     # 3. The LLM's string output goes to the JSON parser.
#     chain = prompt_template | llm | json_parser
#     return chain


# # --- 3. Non-LLM Agents (Simple Python Parsers) ---
# # These agents are fast and free as they don't require an LLM call.

# def pronoun_ratio_agent(transcript: str) -> AgentScore:
#     """A simple parser to calculate the we/(we+I) ratio as a proxy for team-orientation."""
#     text = transcript.lower()
#     # Count variations of "I" and "we"
#     i_count = text.count(" i ") + text.count(" i'm ") + text.count(" i've ") + text.count(" my ")
#     we_count = text.count(" we ") + text.count(" we're ") + text.count(" we've ") + text.count(" our ")
    
#     # Avoid division by zero
#     total_pronouns = i_count + we_count
#     if total_pronouns == 0:
#         ratio = 0.5  # Neutral if no relevant pronouns are found
#     else:
#         ratio = we_count / total_pronouns

#     # Scale the ratio (0.0 to 1.0) to a score from 0 to 10
#     score = int(ratio * 10)
#     evidence = f"'we' variants: {we_count}, 'I' variants: {i_count}"
    
#     return AgentScore(agent_name="PronounRatioAgent", score=score, evidence=evidence)

# def i_dont_know_agent(transcript: str) -> AgentScore:
#     """Flags if the candidate comfortably expresses uncertainty."""
#     text = transcript.lower()
#     if "i don't know" in text or "i am not sure" in text or "i had to find out" in text:
#         score = 10 # High score for intellectual honesty
#         evidence = "Candidate expressed uncertainty, showing intellectual honesty."
#     else:
#         score = 0
#         evidence = "No explicit phrases of uncertainty were found."
        
#     return AgentScore(agent_name="IDontKnowAgent", score=score, evidence=evidence)














# core/agents.py
import os
import json
from typing import Dict, Any
from openai import OpenAI
import toml

from .state import AgentScore

# ──────────────────────────────────────────────────────────────────────────────
# Client selection (OpenAI or OpenRouter)
# ──────────────────────────────────────────────────────────────────────────────
def get_llm_client() -> OpenAI:
    """
    Use OpenAI if OPENAI_API_KEY is set; else fall back to OpenRouter.
    """
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    if openai_key:
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        return OpenAI(api_key=openai_key, base_url=base_url)

    if openrouter_key:
        base_url = os.getenv("OPENROUTER_BASE_URL", "").strip() or "https://openrouter.ai/api/v1"
        default_headers = {
            "HTTP-Referer": os.getenv("OR_HTTP_REFERER", "http://localhost"),
            "X-Title": os.getenv("OR_X_TITLE", "Humility-4"),
        }
        return OpenAI(api_key=openrouter_key, base_url=base_url, default_headers=default_headers)

    raise RuntimeError("No LLM API key found. Set OPENAI_API_KEY or OPENROUTER_API_KEY.")

def get_llm_model_name() -> str:
    return (
        os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("OPENROUTER_MODEL")
        or "gpt-4o-mini"   # default fast model; override via env
    )

# ──────────────────────────────────────────────────────────────────────────────
# Prompts loader
# ──────────────────────────────────────────────────────────────────────────────
_prompts = None
def _load_prompts() -> Dict[str, Any]:
    global _prompts
    if _prompts is None:
        with open("core/prompts.toml", "r", encoding="utf-8") as f:
            _prompts = toml.load(f)
        print("Prompts loaded successfully from core/prompts.toml")
    return _prompts

class _AsyncWrapper:
    def __init__(self, fn):
        self._fn = fn
    async def ainvoke(self, payload: Dict[str, Any]):
        return await self._fn(payload)

# ──────────────────────────────────────────────────────────────────────────────
# Agent factory (LLM-based)
# ──────────────────────────────────────────────────────────────────────────────
def create_analysis_agent(agent_name: str):
    prompts = _load_prompts()
    general = prompts.get("General", {})
    fmt = general.get("json_format_instructions", "")
    section = prompts.get(agent_name, None)
    if not section or not section.get("system_prompt"):
        raise RuntimeError(f"No prompt found for agent '{agent_name}' in core/prompts.toml")

    system_prompt = section["system_prompt"].strip()
    model = get_llm_model_name()
    client = get_llm_client()

    max_chars = int(os.getenv("LLM_MAX_CHARS", "1500"))
    timeout_sec = float(os.getenv("LLM_TIMEOUT_SEC", "20"))

    async def _run(payload: Dict[str, Any]):
        transcript = payload.get("transcript") or payload.get("input") or payload.get("text") or payload.get("response") or ""
        if not transcript:
            return {"score": 0, "evidence": "Empty transcript."}

        t = transcript[-max_chars:]  # truncate for speed

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt + "\n\n" + fmt},
                {"role": "user", "content": t},
            ],
            temperature=0.2,
            max_tokens=180,
            timeout=timeout_sec,
        )

        content = resp.choices[0].message.content.strip()
        content = content.strip().strip("`")
        try:
            obj = json.loads(content)
        except Exception:
            start = content.find("{"); end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    obj = json.loads(content[start:end+1])
                except Exception:
                    obj = {"score": 0, "evidence": "Could not parse model JSON."}
            else:
                obj = {"score": 0, "evidence": "Model did not return JSON."}

        score = obj.get("score", 0)
        ev = obj.get("evidence", "No evidence.")
        return {"score": score, "evidence": ev}

    return _AsyncWrapper(_run)

# ──────────────────────────────────────────────────────────────────────────────
# Non‑LLM parser agents (fast)
# ──────────────────────────────────────────────────────────────────────────────
def pronoun_ratio_agent(transcript: str) -> AgentScore:
    text = " " + (transcript or "").lower() + " "
    i_count = sum(text.count(k) for k in [" i ", " i'm ", " i've ", " my "])
    we_count = sum(text.count(k) for k in [" we ", " we're ", " we've ", " our "])
    total = i_count + we_count
    ratio = 0.5 if total == 0 else we_count / total
    score = int(ratio * 10)
    evidence = f"'we' variants: {we_count}, 'I' variants: {i_count}"
    return AgentScore(agent_name="PronounRatioAgent", score=score, evidence=evidence)

def i_dont_know_agent(transcript: str) -> AgentScore:
    text = (transcript or "").lower()
    if any(p in text for p in ["i don't know", "i am not sure", "i had to find out"]):
        score = 10
        evidence = "Candidate expressed uncertainty, showing intellectual honesty."
    else:
        score = 0
        evidence = "No explicit phrases of uncertainty were found."
    return AgentScore(agent_name="IDontKnowAgent", score=score, evidence=evidence)
