# """Humility analysis agent"""
# import asyncio

# def analyze_humility(transcript: str) -> tuple[float, str]:
#     """
#     Analyze humility indicators in the transcript.
#     Returns a tuple of (score, evidence)
#     """
#     humble_phrases = [
#         "i could be wrong", "i might be mistaken", "in my opinion",
#         "from my perspective", "i think", "i believe", "i feel",
#         "i learned", "i grew", "i improved", "i understand",
#         "i appreciate", "thank you", "grateful", "acknowledge",
#         "recognize", "admit", "own up to", "take responsibility"
#     ]
    
#     humble_indicators = [
#         "mistake", "learn", "grow", "improve", "feedback",
#         "help", "support", "team", "collaborat", "together"
#     ]
    
#     score = 0
#     evidence = []
    
#     # Check for humble phrases
#     for phrase in humble_phrases:
#         if phrase in transcript.lower():
#             score += 0.5
#             evidence.append(f"Found humble phrase: '{phrase}'")
    
#     # Check for humble indicators
#     for indicator in humble_indicators:
#         count = transcript.lower().count(indicator.lower())
#         if count > 0:
#             score += 0.3 * count
#             evidence.append(f"Found {count} instance(s) of '{indicator}'")
    
#     # Cap the score at 10
#     score = min(10.0, round(score, 1))
    
#     # If no evidence found, provide a default message
#     if not evidence:
#         evidence.append("No strong indicators of humility found in the response.")
    
#     return score, "; ".join(evidence)






# backend/agents/humility_agent.py
import re
from typing import Tuple

HEDGE_WORDS = {"i think", "i feel", "i believe", "from my perspective", "in my opinion", "could be", "might be"}
OWNERSHIP = {
    r"\bi was wrong\b", r"\bi made a mistake\b", r"\bi (was|am) at fault\b", r"\bi take responsibility\b",
    r"\bi (own|owned) (it|the mistake)\b", r"\bi (acknowledge|admit)(ted)?\b"
}
CREDIT = {
    r"\bthanks? to (the )?team\b", r"\bwe\b.*\bachieve(d)?\b", r"\bshared credit\b", r"\bi couldn'?t have done it alone\b",
    r"\bappreciat(e|ed) (my )?team\b", r"\bwith support of\b"
}
BRAG_HEAVY = {
    r"\bsingle[- ]handedly\b", r"\bworld[- ]class\b", r"\bthe best\b", r"\bperfect\b", r"\bflawless\b",
    r"\b(i am|i'm) (the )?best\b", r"\bgenius\b"
}

async def analyze_humility(transcript: str) -> Tuple[float, str]:
    """
    Scoring pillars for humility in HR context:
    + Hedge/softeners
    + Ownership of mistakes
    + Sharing credit / team orientation
    - Heavy bragging / superlatives
    + 'We' vs 'I' balance
    """
    if not transcript or not transcript.strip():
        return 0.0, "No text provided"

    t = transcript.lower()

    # hedge/softener
    hedge_score = sum(1 for h in HEDGE_WORDS if h in t) * 0.6

    # ownership (regex)
    own_score = 0.0
    own_hits = []
    for patt in OWNERSHIP:
        m = re.search(patt, t)
        if m:
            own_score += 1.2
            own_hits.append(patt)

    # credit/team
    credit_score = 0.0
    credit_hits = []
    for patt in CREDIT:
        m = re.search(patt, t)
        if m:
            credit_score += 1.0
            credit_hits.append(patt)

    # bragging penalty
    brag_pen = 0.0
    brag_hits = []
    for patt in BRAG_HEAVY:
        m = re.search(patt, t)
        if m:
            brag_pen += 1.5
            brag_hits.append(patt)

    # pronoun balance
    i_count = len(re.findall(r"\bi\b", t))
    we_count = len(re.findall(r"\bwe\b", t))
    # encourage balanced collaboration; strongly "I" heavy reduces humility, "we" adds a little
    pronoun_adj = 0.0
    if we_count >= 1:
        pronoun_adj += min(1.0, 0.2 * we_count)
    if i_count > we_count * 3 and i_count >= 5:
        pronoun_adj -= 1.0  # too self‑centric

    raw = hedge_score + own_score + credit_score + pronoun_adj - brag_pen
    # normalize to 0–10
    score = max(0.0, min(10.0, 5.0 + raw))  # center on 5 and shift by raw

    evidence = []
    if hedge_score > 0:
        evidence.append("Used hedging/softeners")
    if own_hits:
        evidence.append(f"Ownership: {', '.join(h.strip('^$') for h in own_hits)}")
    if credit_hits:
        evidence.append(f"Shared credit/team orientation: {', '.join(h.strip('^$') for h in credit_hits)}")
    if brag_hits:
        evidence.append(f"Bragging/superlatives: {', '.join(h.strip('^$') for h in brag_hits)}")
    if we_count or i_count:
        evidence.append(f"Pronouns — I:{i_count}, We:{we_count}")

    return round(score, 1), (" ; ".join(evidence) if evidence else "No strong humility indicators found")
