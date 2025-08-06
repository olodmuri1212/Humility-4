# core/agents.py

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

from .config import NON_REASONING_API_KEY, OPENROUTER_API_BASE, PROMPTS
from .state import AgentScore

# --- 1. Reusable LLM and Parser Setup ---

# Initialize the LLM client to communicate with OpenRouter's API.
# We configure it once here and reuse it for all agents.
llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    openai_api_key=NON_REASONING_API_KEY,
    openai_api_base=OPENROUTER_API_BASE,
    temperature=0.1,  # Low temperature for consistent, analytical output
    max_tokens=150,
    model_kwargs={
        # This header is recommended by OpenRouter for tracking.
        "extra_headers": {"HTTP-Referer": "http://localhost:8501"}
    },
)

# Initialize a JSON parser that will try to shape the LLM's output
# into our Pydantic 'AgentScore' model.
json_parser = JsonOutputParser(pydantic_object=AgentScore)


# --- 2. Agent Factory (for LLM-based agents) ---

def create_analysis_agent(agent_name: str):
    """
    A factory function to dynamically create a LangChain agent (a "chain")
    for any given humility trait defined in prompts.toml.

    Args:
        agent_name: The name of the agent, which must match a section in prompts.toml.

    Returns:
        A LangChain runnable sequence (a "chain") ready to be invoked.
    """
    if agent_name not in PROMPTS:
        raise ValueError(f"Prompt for agent '{agent_name}' not found in prompts.toml")

    # Combine the agent's specific system prompt with the general JSON formatting instructions.
    # This is a crucial step for reliable JSON output.
    prompt_text = (
        PROMPTS[agent_name]["system_prompt"]
        + "\n\n"
        + PROMPTS["General"]["json_format_instructions"]
        + "\n\nCandidate's response: {transcript}"
    )

    prompt_template = ChatPromptTemplate.from_template(prompt_text)

    # Create the chain by piping the components together:
    # 1. The input dictionary goes into the prompt template.
    # 2. The formatted prompt goes to the LLM.
    # 3. The LLM's string output goes to the JSON parser.
    chain = prompt_template | llm | json_parser
    return chain


# --- 3. Non-LLM Agents (Simple Python Parsers) ---
# These agents are fast and free as they don't require an LLM call.

def pronoun_ratio_agent(transcript: str) -> AgentScore:
    """A simple parser to calculate the we/(we+I) ratio as a proxy for team-orientation."""
    text = transcript.lower()
    # Count variations of "I" and "we"
    i_count = text.count(" i ") + text.count(" i'm ") + text.count(" i've ") + text.count(" my ")
    we_count = text.count(" we ") + text.count(" we're ") + text.count(" we've ") + text.count(" our ")
    
    # Avoid division by zero
    total_pronouns = i_count + we_count
    if total_pronouns == 0:
        ratio = 0.5  # Neutral if no relevant pronouns are found
    else:
        ratio = we_count / total_pronouns

    # Scale the ratio (0.0 to 1.0) to a score from 0 to 10
    score = int(ratio * 10)
    evidence = f"'we' variants: {we_count}, 'I' variants: {i_count}"
    
    return AgentScore(agent_name="PronounRatioAgent", score=score, evidence=evidence)

def i_dont_know_agent(transcript: str) -> AgentScore:
    """Flags if the candidate comfortably expresses uncertainty."""
    text = transcript.lower()
    if "i don't know" in text or "i am not sure" in text or "i had to find out" in text:
        score = 10 # High score for intellectual honesty
        evidence = "Candidate expressed uncertainty, showing intellectual honesty."
    else:
        score = 0
        evidence = "No explicit phrases of uncertainty were found."
        
    return AgentScore(agent_name="IDontKnowAgent", score=score, evidence=evidence)
