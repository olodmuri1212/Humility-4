# """Learning mindset analysis agent"""

# def analyze_learning(transcript: str) -> tuple[float, str]:
#     """
#     Analyze learning mindset indicators in the transcript.
#     Returns a tuple of (score, evidence)
#     """
#     learning_indicators = [
#         "learn", "study", "research", "read", "explore", "discover",
#         "understand", "figure out", "find out", "gain insight", "grow",
#         "develop", "improve", "enhance", "progress", "advance", "master",
#         "skill", "knowledge", "expertise", "competence", "ability"
#     ]
    
#     score = 0
#     evidence = []
    
#     # Check for learning indicators
#     for indicator in learning_indicators:
#         count = transcript.lower().count(indicator.lower())
#         if count > 0:
#             score += 0.5 * count
#             evidence.append(f"Found learning indicator: '{indicator}'")
    
#     # Check for past learning experiences
#     learning_experience_phrases = [
#         "i learned", "i discovered", "i found out", "i realized",
#         "came to understand", "was able to learn", "expanded my knowledge"
#     ]
    
#     for phrase in learning_experience_phrases:
#         if phrase in transcript.lower():
#             score += 1.0
#             evidence.append(f"Mentioned learning experience: '{phrase}'")
    
#     # Cap the score at 10
#     score = min(10.0, round(score, 1))
    
#     if not evidence:
#         evidence.append("No strong indicators of a learning mindset found in the response.")
    
#     return score, "; ".join(evidence)










# backend/agents/learning_agent.py
import re
from typing import Tuple

LEARN_TERMS = {
    "learn", "upskill", "study", "research", "read", "explore", "discover",
    "understand", "figure out", "find out", "gain insight", "grow", "develop",
    "improve", "enhance", "progress", "advance", "master"
}
APPLIED_TERMS = {
    "implemented", "applied", "experimented", "prototyped", "piloted",
    "iterated", "refactored", "automated", "optimized"
}
RESOURCE_TERMS = {"course", "certification", "mentor", "documentation", "tutorial", "workshop", "bootcamp", "training"}
RETRO_TERMS = {"retrospective", "postmortem", "lessons learned", "root cause", "rca"}

def _count_occurrences(text: str, words: set[str]) -> int:
    cnt = 0
    for w in words:
        cnt += text.count(w)
    return cnt

async def analyze_learning(transcript: str) -> Tuple[float, str]:
    """
    Scores learning mindset with emphasis on *applied learning* in HR context:
    + mentions of learning/growth
    + use of resources/mentors
    + application (implemented, prototyped, iterated)
    + structured reflection (retro/lessons learned)
    """
    if not transcript or not transcript.strip():
        return 0.0, "No text provided"

    t = transcript.lower()

    base = _count_occurrences(t, LEARN_TERMS) * 0.6
    applied = _count_occurrences(t, APPLIED_TERMS) * 0.9
    resources = _count_occurrences(t, RESOURCE_TERMS) * 0.7
    retro = _count_occurrences(t, RETRO_TERMS) * 1.0

    # explicit phrases of past learning experience
    past_learning = 0.0
    pl_hits = []
    for patt in [r"\bi learned\b", r"\bi discovered\b", r"\bi realized\b", r"\bwhat i learned\b", r"\blessons learned\b"]:
        m = re.search(patt, t)
        if m:
            past_learning += 1.2
            pl_hits.append(patt)

    raw = base + applied + resources + retro + past_learning
    score = max(0.0, min(10.0, 3.0 + raw))  # anchor at 3 and add raw

    evidence = []
    if base > 0:
        evidence.append("Growth/learning terms mentioned")
    if applied > 0:
        evidence.append("Applied learning (implemented/prototyped/etc.)")
    if resources > 0:
        evidence.append("Used resources/mentors/courses")
    if retro > 0:
        evidence.append("Structured reflection (retro/RCA/lessons learned)")
    if pl_hits:
        evidence.append("Past learning experiences described")

    return round(score, 1), (" ; ".join(evidence) if evidence else "No strong indicators of a learning mindset found")
