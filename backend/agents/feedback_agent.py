# import re
# from typing import Tuple


# async def analyze_feedback_seeking(text: str) -> Tuple[float, str]:
#     """
#     Looks for phrases asking for feedback, advice, or input.
#     Returns 0–10 scale.
#     """
#     if not text or not text.strip():
#         return 0.0, "No text provided"

#     indicators = [
#         r"\bfeedback\b",
#         r"\bsuggestions?\b",
#         r"\badvice\b",
#         r"\binput\b",
#         r"\bwhat do you think\b",
#         r"\bhow can i improve\b",
#         r"\bconstructive criticism\b",
#         r"\bany thoughts\b",
#         r"\bany tips\b"
#     ]

#     found = []
#     lower = text.lower()
#     for patt in indicators:
#         if re.search(patt, lower):
#             found.append(patt.strip(r"\b"))

#     score = min(10.0, len(found) * 2.0)
#     if found:
#         return score, f"Found feedback‐seeking indicators: {', '.join(found)}"
#     else:
#         return 0.0, "No feedback‐seeking indicators found"






# ## backend/agents/feedback_agent.py #take2
# import re
# from typing import Tuple, List

# # Helper: safely find matches with word boundaries and basic negation handling
# def _count_patterns(text: str, patterns: List[re.Pattern]) -> int:
#     return sum(1 for p in patterns if p.search(text))

# def _negated(text: str, span_start: int, window: int = 5) -> bool:
#     # crude negation: "not", "no", "never" within 5 tokens before the match
#     left = text[:span_start]
#     tokens = re.findall(r"\w+", left.lower())[-window:]
#     return any(t in {"not", "no", "never", "don't", "dont", "isn't", "isnt", "won't", "wont"} for t in tokens)

# async def analyze_feedback_seeking(text: str) -> Tuple[float, str]:
#     """
#     Detects explicit/implicit feedback‑seeking aligned to HR interviews:
#     - explicit asks (feedback, suggestions, advice, input, thoughts, tips)
#     - openness phrases (keen to, open to, happy to, appreciate feedback)
#     - closing invites (anything I can improve / what would you change)
#     Penalizes: feedback‑avoidance or defensiveness.
#     Returns: (score [0..10], evidence string)
#     """
#     if not text or not text.strip():
#         return 0.0, "No text provided"

#     lower = text.lower()

#     explicit = [
#         r"\bfeedback\b",
#         r"\bsuggestions?\b",
#         r"\badvice\b",
#         r"\binput\b",
#         r"\bwhat do you think\b",
#         r"\bhow can i improve\b",
#         r"\bconstructive criticism\b",
#         r"\bany thoughts\b",
#         r"\bany tips\b",
#         r"\bplease (share|give) (your )?feedback\b",
#         r"\bwould love (your )?(thoughts|feedback|advice)\b",
#     ]
#     openness = [
#         r"\bopen to (feedback|suggestions|input)\b",
#         r"\bhappy to (receive|take) feedback\b",
#         r"\bkeen to (learn|improve|hear your thoughts)\b",
#         r"\bappreciate (your )?feedback\b",
#         r"\b(eager|willing) to improve\b",
#     ]
#     closers = [
#         r"\banything (i|we) can improve\b",
#         r"\bwhat would you (change|suggest)\b",
#         r"\bhow (could|can) i do this better\b",
#     ]
#     avoidance = [
#         r"\bdon't need feedback\b",
#         r"\bno feedback (needed|required)\b",
#         r"\bnot looking for feedback\b",
#         r"\bi already know\b",
#         r"\b(there was|is) nothing to improve\b",
#     ]

#     exp_re = [re.compile(p) for p in explicit]
#     open_re = [re.compile(p) for p in openness]
#     clos_re = [re.compile(p) for p in closers]
#     avoid_re = [re.compile(p) for p in avoidance]

#     # count signals
#     exp_hits = []
#     for p in exp_re:
#         m = p.search(lower)
#         if m and not _negated(lower, m.start()):
#             exp_hits.append(p.pattern)

#     open_hits = []
#     for p in open_re:
#         m = p.search(lower)
#         if m and not _negated(lower, m.start()):
#             open_hits.append(p.pattern)

#     close_hits = []
#     for p in clos_re:
#         m = p.search(lower)
#         if m and not _negated(lower, m.start()):
#             close_hits.append(p.pattern)

#     avoid_hits = [p.pattern for p in avoid_re if p.search(lower)]

#     # question‑mark as soft signal (only if sentence contains “feedback/advise” tokens)
#     qm_bonus = 0.0
#     if "?" in text and (("feedback" in lower) or ("advice" in lower) or ("suggest" in lower)):
#         qm_bonus = 0.5

#     # scoring
#     pos = 2.0 * len(exp_hits) + 1.0 * len(open_hits) + 1.0 * len(close_hits) + qm_bonus
#     neg = 2.5 * len(avoid_hits)

#     raw = max(0.0, min(10.0, pos * 1.4 - neg))  # amplify positives slightly but cap

#     evidence_parts = []
#     if exp_hits:
#         evidence_parts.append(f"Explicit asks: {', '.join([h.strip('^$') for h in exp_hits])}")
#     if open_hits:
#         evidence_parts.append(f"Openness: {', '.join([h.strip('^$') for h in open_hits])}")
#     if close_hits:
#         evidence_parts.append(f"Closing invites: {', '.join([h.strip('^$') for h in close_hits])}")
#     if avoid_hits:
#         evidence_parts.append(f"Avoidance: {', '.join([h.strip('^$') for h in avoid_hits])}")
#     if qm_bonus > 0:
#         evidence_parts.append("Used a question to invite feedback")

#     evidence = " ; ".join(evidence_parts) if evidence_parts else "No feedback‑seeking indicators found"
#     return round(raw, 1), evidence

















##take2
# backend/agents/feedback_agent.py
import re
from typing import List

def _has_quant(text: str) -> bool:
    return bool(re.search(r"\b\d+(\.\d+)?\b|%|percent|minutes?|hours?|days?|weeks?|months?|kpis?|roi|revenue|cost|time to", text.lower()))

def _pronoun_balance(text: str) -> str:
    t = text.lower()
    i_count = len(re.findall(r"\bi\b", t))
    we_count = len(re.findall(r"\bwe\b", t))
    if we_count == 0 and i_count > 4:
        return "Shift some 'I' statements to 'we' and share credit to show collaboration."
    if we_count < i_count // 3 and i_count >= 6:
        return "Balance ‘I’ with ‘we’; highlight the team’s role where appropriate."
    return ""

def _has_ownership(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in ["i took responsibility", "i owned", "i take responsibility", "my mistake", "i was wrong"])

def _has_rca(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in ["rca", "root cause", "postmortem", "retro", "retrospective"])

def _has_prevention(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in ["checklist", "added tests", "monitoring", "alert", "guardrail", "prevention", "prevent"])

def _has_feedback_invite(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in ["would love your feedback", "keen to hear feedback", "what could i improve", "any suggestions", "any advice", "any thoughts"])

def _has_learning_loop(text: str) -> bool:
    t = text.lower()
    # very light STAR/loop signal: learn/apply/result
    learn = any(w in t for w in ["learn", "study", "research", "course", "mentor", "docs", "documentation"])
    apply_ = any(w in t for w in ["applied", "implemented", "piloted", "prototyped", "iterated", "refactored"])
    result = _has_quant(t) or any(w in t for w in ["improved", "reduced", "increased", "faster", "slower", "better"])
    return sum([learn, apply_, result]) >= 2

async def analyze_feedback(text: str) -> str:
    """
    Generate concise, constructive feedback for the *given answer*.
    No numeric score is returned. Output is a short paragraph.
    """
    if not text or not text.strip():
        return "Consider adding a clear example, what you did, and what improved as a result."

    suggestions: List[str] = []

    pb = _pronoun_balance(text)
    if pb: suggestions.append(pb)

    if not _has_quant(text):
        suggestions.append("Quantify impact (e.g., ‘reduced turnaround by 30%’, ‘cut 4 hours/week’).")

    if not _has_learning_loop(text):
        suggestions.append("Show a learning loop: what you learned, how you applied it, and the outcome.")

    if not _has_ownership(text):
        suggestions.append("If relevant, state ownership explicitly (e.g., ‘I took responsibility’).")

    if not _has_rca(text):
        suggestions.append("Briefly mention analysis (e.g., ‘ran a quick RCA’ or ‘root cause analysis’).")

    if not _has_prevention(text):
        suggestions.append("Close with prevention: tests, checklist, monitoring, or process tweak.")

    if not _has_feedback_invite(text):
        suggestions.append("Invite feedback with a line like ‘Would value your feedback on any blind spots.’")

    # de-duplicate and keep it concise
    out = []
    seen = set()
    for s in suggestions:
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    if not out:
        return "Good structure and clarity. Keep highlighting collaboration, measurable outcomes, and what you learned."

    # Turn into a compact paragraph
    return " ".join(out)


async def analyze_feedback_seeking(text: str):
    """
    Backwards-compatible wrapper.
    Returns a tuple so old callers don't crash:
      (score_placeholder, feedback_text)
    """
    from .feedback_agent import analyze_feedback  # self-import is fine here
    fb = await analyze_feedback(text)
    return 0.0, fb