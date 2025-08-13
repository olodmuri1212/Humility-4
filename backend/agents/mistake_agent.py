# import re
# from typing import Tuple


# async def analyze_mistake_handling(text: str) -> Tuple[float, str]:
#     """
#     Checks for positive vs. negative language around mistakes.
#     Positive increases score, negative decreases. Returns 0–10.
#     """
#     if not text or not text.strip():
#         return 0.0, "No text provided"

#     positive = [
#         r"\blearn(ed)? from\b",
#         r"\bgrow(n)? from\b",
#         r"\bimprove(d)? after\b",
#         r"\btook responsibility\b",
#         r"\badmit(ted)?\b",
#         r"\backnowledge(d)?\b"
#     ]
#     negative = [
#         r"\bblame\b",
#         r"\bexcuse(s)?\b",
#         r"\bnot my fault\b",
#         r"\bthey made me\b",
#         r"\bhad to\b",
#         r"\bno choice\b",
#         r"\beveryone else\b"
#     ]

#     pos_found = []
#     neg_found = []
#     lower = text.lower()

#     for patt in positive:
#         if re.search(patt, lower):
#             pos_found.append(patt.strip(r"\b"))

#     for patt in negative:
#         if re.search(patt, lower):
#             neg_found.append(patt.strip(r"\b"))

#     # start at 0, +1 per positive, -1.5 per negative
#     score_val = len(pos_found) * 1.0 - len(neg_found) * 1.5
#     # scale to 0–10 by factor 2 (so max ~10)
#     raw = max(0.0, min(10.0, score_val * 2.0))

#     evidence_parts = []
#     if pos_found:
#         evidence_parts.append("positive: " + ", ".join(pos_found))
#     if neg_found:
#         evidence_parts.append("negative: " + ", ".join(neg_found))

#     evidence = " ; ".join(evidence_parts) if evidence_parts else "No clear indicators"
#     return raw, f"Mistake handling: {evidence}"



# backend/agents/mistake_agent.py
import re
from typing import Tuple

POSITIVE = [
    r"\bi took responsibility\b",
    r"\bi (owned|own) (the )?(issue|mistake|error)\b",
    r"\bi (admit|admitted|acknowledge|acknowledged)\b",
    r"\bi learned from\b",
    r"\bi (grew|improved) (from|after)\b",
    r"\bwe ran (a )?(postmortem|retro|retrospective)\b",
    r"\bdid (an )?rca\b",
    r"\b(root cause|prevent(ion)? plan|action items)\b",
    r"\badded (tests|monitoring|alerts)\b",
    r"\bimplemented (a )?fix\b",
]
NEGATIVE = [
    r"\bnot my fault\b",
    r"\b(i|we) (were|was) forced\b",
    r"\bthey made me\b",
    r"\b(blame|blamed)\b",
    r"\bno choice\b",
    r"\b(it|this) (was|is) (someone|somebody|others?)'?s fault\b",
]

def _has_negation(text: str, start: int, window_tokens: int = 5) -> bool:
    left = text[:start]
    toks = re.findall(r"\w+", left)[-window_tokens:]
    return any(t.lower() in {"not","no","never"} for t in toks)

async def analyze_mistake_handling(text: str) -> Tuple[float, str]:
    """
    HR‑grade mistake handling:
    + ownership, RCA, fix/prevention, learning
    - blame‑shifting / excuses
    """
    if not text or not text.strip():
        return 0.0, "No text provided"

    t = text.lower()

    pos_hits, neg_hits = [], []
    pos_score = 0.0
    for patt in POSITIVE:
        m = re.search(patt, t)
        if m and not _has_negation(t, m.start()):
            pos_hits.append(patt)
            # heavier weight for RCA/prevention
            if any(k in patt for k in ["postmortem", "retro", "rca", "root cause", "prevent"]):
                pos_score += 1.5
            else:
                pos_score += 1.0

    neg_score = 0.0
    for patt in NEGATIVE:
        m = re.search(patt, t)
        if m:
            neg_hits.append(patt)
            neg_score += 1.3

    # Also watch for "but" excuse patterns (“I did X, but …”)
    excuse_pen = 0.0
    if re.search(r"\b(i took responsibility|i owned it)\b.*\bbut\b", t):
        excuse_pen += 1.0

    raw = 4.0 + pos_score * 1.2 - (neg_score + excuse_pen) * 1.4  # base 4, push by signals
    score = max(0.0, min(10.0, raw))

    evidence = []
    if pos_hits:
        evidence.append(f"Positive: {', '.join(h.strip('^$') for h in pos_hits)}")
    if neg_hits:
        evidence.append(f"Negative: {', '.join(h.strip('^$') for h in neg_hits)}")
    if excuse_pen > 0:
        evidence.append("Responsibility undercut by 'but' clause")

    return round(score, 1), (" ; ".join(evidence) if evidence else "No clear indicators")
