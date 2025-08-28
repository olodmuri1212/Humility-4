# #original code
# # backend/agent_manager.py

# import sys
# import os
# from typing import List, NamedTuple , Tuple
# import asyncio
# from backend.agents.humility_agent import analyze_humility 
# from backend.agents.learning_agent import analyze_learning as analyze_learning_mindset
# from backend.agents.feedback_agent import analyze_feedback_seeking
# from backend.agents.mistake_agent import analyze_mistake_handling
# # Add the project root to the Python path
# # This allows us to import modules from 'core' and 'services'
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from core.agents import (
#     create_analysis_agent,
#     pronoun_ratio_agent,
#     i_dont_know_agent
# )
# from core.state import AgentScore
# class AnalysisResult(NamedTuple):          
#     """Container for the four agent outputs."""                                 
#     humility_score: float                  
#     humility_evidence: str                 
#     learning_score: float                  
#     learning_evidence: str                 
#     feedback_score: float                  
#     feedback_evidence: str                 
#     mistakes_score: float                  
#     mistakes_evidence: str 
# # These are the LLM-based agents that will be run on each transcript.
# # You can easily add or remove agents from this list to change the analysis.
# LLM_AGENTS_TO_RUN = [
#     "AdmitMistakeAgent",
#     "MindChangeAgent",
#     "ShareCreditAgent",
#     "LearnerMindsetAgent",
#     "BragFlagAgent",
#     "BlameShiftAgent",
#     "KnowItAllAgent",
#     "FeedbackAcceptanceAgent",
#     "SupportGrowthAgent",
#     "PraiseHandlingAgent" # This would be triggered conditionally in a more advanced setup
# ]

# async def run_analysis_pipeline(transcript: str) -> List[AgentScore]:
#     """
#     Runs all relevant analysis agents on a given transcript.
#     It runs simple parser-based agents and complex LLM-based agents in parallel.
    
#     Args:
#         transcript: The text of the candidate's response.

#     Returns:
#         A list of AgentScore objects from all agents.
#     """
#     all_scores: List[AgentScore] = []
#     print("--- Starting Analysis Pipeline ---")

#     # --- 1. Run Simple Parser Agents (Fast, No LLM calls) ---
#     # These are synchronous and fast, so we run them directly.
#     try:
#         all_scores.append(pronoun_ratio_agent(transcript))
#         all_scores.append(i_dont_know_agent(transcript))
#         print("Executed simple parser agents.")
#     except Exception as e:
#         print(f"Error in simple parser agents: {e}")
        
#     # --- 2. Run LLM-based Agents in Parallel ---
#     llm_tasks = []
#     for agent_name in LLM_AGENTS_TO_RUN:
#         try:
#             # Create the specific agent chain from the factory
#             agent_chain = create_analysis_agent(agent_name)
#             # Create an async task for each agent invocation. This schedules them to run concurrently.
#             task = agent_chain.ainvoke({"transcript": transcript})
#             llm_tasks.append((agent_name, task))
#         except Exception as e:
#             print(f"Error creating agent {agent_name}: {e}")

#     print(f"Scheduled {len(llm_tasks)} LLM agents to run in parallel.")

#     # Await all scheduled LLM agent tasks. `asyncio.gather` runs them concurrently.
#     # `return_exceptions=True` prevents one failed agent from stopping all others.
#     llm_results = await asyncio.gather(*(task for _, task in llm_tasks), return_exceptions=True)
    
#     # --- 3. Process and Aggregate Results ---
#     for i, result in enumerate(llm_results):
#         agent_name = llm_tasks[i][0]
#         if isinstance(result, Exception):
#             print(f"Agent '{agent_name}' failed with an error: {result}")
#             # Add a default/error score so the UI doesn't break
#             all_scores.append(AgentScore(agent_name=agent_name, score=0, evidence="Agent execution failed."))
#         elif result and isinstance(result, dict):
#             # Convert the dictionary from the JSON parser into an AgentScore object.
#             # The 'agent_name' is assigned here, not in the prompt chain.
#             score_obj = AgentScore(
#                 agent_name=agent_name,
#                 score=result.get('score', 0),
#                 evidence=result.get('evidence', 'No evidence provided.')
#             )
#             all_scores.append(score_obj)
#             print(f"Successfully processed result from agent: {agent_name}")
#         else:
#             print(f"Agent '{agent_name}' returned an empty or invalid result: {result}")
#             all_scores.append(AgentScore(agent_name=agent_name, score=0, evidence="Agent returned no output."))
            
#     print(f"--- Analysis Pipeline Complete. Aggregated {len(all_scores)} scores. ---")
#     return all_scores



# from .agents.humility_agent import analyze_humility
# from .agents.learning_agent import analyze_learning
# from .agents.feedback_agent import analyze_feedback_seeking
# from .agents.mistake_agent import analyze_mistake_handling





















# #2nd
# """
# Simple agent manager for humility analysis
# """

# import asyncio
# from typing import List, Dict, Any, NamedTuple, Tuple
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class AnalysisResult(NamedTuple):
#     """Container for analysis results from all agents"""
#     humility_score: float
#     humility_evidence: str
#     learning_score: float
#     learning_evidence: str
#     feedback_score: float
#     feedback_evidence: str
#     mistakes_score: float
#     mistakes_evidence: str

#     def to_dict(self) -> Dict[str, Any]:
#         """Convert the analysis result to a dictionary"""
#         return {
#             'humility': {'score': self.humility_score, 'evidence': self.humility_evidence},
#             'learning': {'score': self.learning_score, 'evidence': self.learning_evidence},
#             'feedback': {'score': self.feedback_score, 'evidence': self.feedback_evidence},
#             'mistakes': {'score': self.mistakes_score, 'evidence': self.mistakes_evidence}
#         }

# def calculate_overall_score(results: Dict[str, Tuple[float, str]]) -> float:
#     """Calculate an overall score from all analysis results"""
#     total = 0.0
#     count = 0
    
#     for score, _ in results.values():
#         total += score
#         count += 1
    
#     return total / count if count > 0 else 0.0

# async def analyze_humility(text: str) -> Tuple[float, str]:
#     """Analyze text for humility indicators"""
#     if not text.strip():
#         return 0.0, "No text provided for analysis"
    
#     # Simple keyword-based analysis
#     humility_indicators = [
#         "i think", "maybe", "perhaps", "in my opinion", 
#         "i believe", "i feel", "i would say", "i'm not sure",
#         "i could be wrong", "correct me if i'm wrong"
#     ]
    
#     score = 0.0
#     evidence = []
    
#     for indicator in humility_indicators:
#         if indicator in text.lower():
#             score += 1.0
#             evidence.append(indicator)
    
#     # Cap score at 10
#     score = min(10.0, score * 2)  # Scale to 0-10 range
    
#     if evidence:
#         return score, f"Found humility indicators: {', '.join(evidence)}"
#     return score, "No strong indicators of humility found"

# async def analyze_learning_mindset(text: str) -> Tuple[float, str]:
#     """Analyze text for learning mindset indicators"""
#     if not text.strip():
#         return 0.0, "No text provided for analysis"
    
#     learning_indicators = [
#         "learned", "discovered", "realized", "understood",
#         "grew", "improved", "developed", "gained insight",
#         "expanded my knowledge", "new perspective"
#     ]
    
#     score = 0.0
#     evidence = []
    
#     for indicator in learning_indicators:
#         if indicator in text.lower():
#             score += 1.0
#             evidence.append(indicator)
    
#     # Cap score at 10
#     score = min(10.0, score * 2)  # Scale to 0-10 range
    
#     if evidence:
#         return score, f"Found learning indicators: {', '.join(evidence)}"
#     return score, "No strong indicators of learning mindset found"

# async def analyze_feedback_seeking(text: str) -> Tuple[float, str]:
#     """Analyze text for feedback seeking behavior"""
#     if not text.strip():
#         return 0.0, "No text provided for analysis"
    
#     feedback_indicators = [
#         "feedback", "suggestions", "advice", "input",
#         "thoughts", "opinion", "what do you think",
#         "how can i improve", "constructive criticism"
#     ]
    
#     score = 0.0
#     evidence = []
    
#     for indicator in feedback_indicators:
#         if indicator in text.lower():
#             score += 1.0
#             evidence.append(indicator)
    
#     # Cap score at 10
#     score = min(10.0, score * 2)  # Scale to 0-10 range
    
#     if evidence:
#         return score, f"Found feedback seeking indicators: {', '.join(evidence)}"
#     return score, "No strong indicators of feedback seeking found"

# async def analyze_mistake_handling(text: str) -> Tuple[float, str]:
#     """Analyze text for how mistakes are handled"""
#     if not text.strip():
#         return 0.0, "No text provided for analysis"
    
#     positive_indicators = [
#         "learned from", "grew from", "improved after",
#         "took responsibility", "admitted", "acknowledged"
#     ]
    
#     negative_indicators = [
#         "blamed", "excuses", "not my fault", "they made me",
#         "had to", "no choice", "everyone else"
#     ]
    
#     score = 5.0  # Start with neutral score
#     evidence = []
    
#     # Check for positive indicators
#     for indicator in positive_indicators:
#         if indicator in text.lower():
#             score += 1.0
#             evidence.append(f"positive: {indicator}")
    
#     # Check for negative indicators
#     for indicator in negative_indicators:
#         if indicator in text.lower():
#             score -= 1.5
#             evidence.append(f"negative: {indicator}")
    
#     # Ensure score is between 0 and 10
#     score = max(0.0, min(10.0, score * 2))  # Scale to 0-10 range
    
#     if evidence:
#         return score, f"Mistake handling indicators: {', '.join(evidence)}"
#     return score, "No clear indicators of mistake handling found"

# async def run_analysis_pipeline(text: str) -> AnalysisResult:
#     """
#     Run all analysis functions in parallel and return combined results
    
#     Args:
#         text: The text to analyze
        
#     Returns:
#         AnalysisResult: Combined results from all analysis functions
#     """
#     if not text.strip():
#         return AnalysisResult(0.0, "No text", 0.0, "No text", 0.0, "No text", 0.0, "No text")
    
#     try:
#         # Run all analysis functions in parallel
#         humility_score, humility_evidence = await analyze_humility(text)
#         learning_score, learning_evidence = await analyze_learning_mindset(text)
#         feedback_score, feedback_evidence = await analyze_feedback_seeking(text)
#         mistakes_score, mistakes_evidence = await analyze_mistake_handling(text)
        
#         return AnalysisResult(
#             humility_score=humility_score,
#             humility_evidence=humility_evidence,
#             learning_score=learning_score,
#             learning_evidence=learning_evidence,
#             feedback_score=feedback_score,
#             feedback_evidence=feedback_evidence,
#             mistakes_score=mistakes_score,
#             mistakes_evidence=mistakes_evidence
#         )
#     except Exception as e:
#         logger.error(f"Error in analysis pipeline: {str(e)}")
#         return AnalysisResult(0.0, "Error", 0.0, "Error", 0.0, "Error", 0.0, "Error")

# # For backward compatibility
# async def analyze_learning(*args, **kwargs):
#     """Alias for analyze_learning_mindset"""
#     return await analyze_learning_mindset(*args, **kwargs)

# async def analyze_mistakes(*args, **kwargs):
#     """Alias for analyze_mistake_handling"""
#     return await analyze_mistake_handling(*args, **kwargs)






















# ##take 5
# import sys
# import os
# from typing import List, NamedTuple , Tuple
# import asyncio
# from backend.agents.humility_agent import analyze_humility 
# from backend.agents.learning_agent import analyze_learning as analyze_learning_mindset
# from backend.agents.feedback_agent import analyze_feedback_seeking
# from backend.agents.mistake_agent import analyze_mistake_handling
# # Add the project root to the Python path
# # This allows us to import modules from 'core' and 'services'
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from core.agents import (
#     create_analysis_agent,
#     pronoun_ratio_agent,
#     i_dont_know_agent
# )
# from core.state import AgentScore
# class AnalysisResult(NamedTuple):          
#     """Container for the four agent outputs."""                                 
#     humility_score: float                  
#     humility_evidence: str                 
#     learning_score: float                  
#     learning_evidence: str                 
#     feedback_score: float                  
#     feedback_evidence: str                 
#     mistakes_score: float                  
#     mistakes_evidence: str 
# # These are the LLM-based agents that will be run on each transcript.
# # You can easily add or remove agents from this list to change the analysis.
# LLM_AGENTS_TO_RUN = [
#     "AdmitMistakeAgent",
#     "MindChangeAgent",
#     "ShareCreditAgent",
#     "LearnerMindsetAgent",
#     "BragFlagAgent",
#     "BlameShiftAgent",
#     "KnowItAllAgent",
#     "FeedbackAcceptanceAgent",
#     "SupportGrowthAgent",
#     "PraiseHandlingAgent" # This would be triggered conditionally in a more advanced setup
# ]

# async def run_analysis_pipeline(transcript: str) -> List[AgentScore]:
#     """
#     Runs all relevant analysis agents on a given transcript.
#     It runs simple parser-based agents and complex LLM-based agents in parallel.
    
#     Args:
#         transcript: The text of the candidate's response.

#     Returns:
#         A list of AgentScore objects from all agents.
#     """
#     all_scores: List[AgentScore] = []
#     print("--- Starting Analysis Pipeline ---")

#     # --- 1. Run Simple Parser Agents (Fast, No LLM calls) ---
#     # These are synchronous and fast, so we run them directly.
#     try:
#         all_scores.append(pronoun_ratio_agent(transcript))
#         all_scores.append(i_dont_know_agent(transcript))
#         print("Executed simple parser agents.")
#     except Exception as e:
#         print(f"Error in simple parser agents: {e}")
        
#     # --- 2. Run LLM-based Agents in Parallel ---
#     llm_tasks = []
#     for agent_name in LLM_AGENTS_TO_RUN:
#         try:
#             # Create the specific agent chain from the factory
#             agent_chain = create_analysis_agent(agent_name)
#             # Create an async task for each agent invocation. This schedules them to run concurrently.
#             task = agent_chain.ainvoke({"transcript": transcript})
#             llm_tasks.append((agent_name, task))
#         except Exception as e:
#             print(f"Error creating agent {agent_name}: {e}")

#     print(f"Scheduled {len(llm_tasks)} LLM agents to run in parallel.")

#     # Await all scheduled LLM agent tasks. `asyncio.gather` runs them concurrently.
#     # `return_exceptions=True` prevents one failed agent from stopping all others.
#     llm_results = await asyncio.gather(*(task for _, task in llm_tasks), return_exceptions=True)

#     for i, result in enumerate(llm_results):
#         agent_name = llm_tasks[i][0]
#         if isinstance(result, Exception):
#             all_scores.append({"agent_name": agent_name, "score": 0, "evidence": "Agent execution failed."})
#         elif result and isinstance(result, dict):
#             all_scores.append({
#                 "agent_name": agent_name,
#                 "score": result.get('score', 0),
#                 "evidence": result.get('evidence', 'No evidence provided.')
#             })
#         else:
#             all_scores.append({"agent_name": agent_name, "score": 0, "evidence": "Agent returned no output."})

#     return all_scores



# # backend/agent_manager.py  (small hardening)
# import sys, os, asyncio
# from typing import List
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from core.agents import create_analysis_agent, pronoun_ratio_agent, i_dont_know_agent
# from core.state import AgentScore

# LLM_AGENTS_TO_RUN = [
#     "AdmitMistakeAgent",
#     "MindChangeAgent",
#     "ShareCreditAgent",
#     "LearnerMindsetAgent",
#     "BragFlagAgent",
#     "BlameShiftAgent",
#     "KnowItAllAgent",
#     "FeedbackAcceptanceAgent",
#     "SupportGrowthAgent",
#     "PraiseHandlingAgent"
# ]

# async def run_analysis_pipeline(transcript: str) -> List[dict]:
#     all_scores: List[dict] = []
#     # simple parser agents (we still compute, even if later hidden in UI/PDF)
#     try:
#         pr = pronoun_ratio_agent(transcript)
#         dk = i_dont_know_agent(transcript)
#         # normalize to dicts
#         all_scores.append({"agent_name": pr.agent_name if hasattr(pr, "agent_name") else "PronounRatioAgent",
#                            "score": float(pr.score if hasattr(pr, "score") else pr.get("score", 0)),
#                            "evidence": getattr(pr, "evidence", "") or pr.get("evidence", "")})
#         all_scores.append({"agent_name": dk.agent_name if hasattr(dk, "agent_name") else "IDontKnowAgent",
#                            "score": float(dk.score if hasattr(dk, "score") else dk.get("score", 0)),
#                            "evidence": getattr(dk, "evidence", "") or dk.get("evidence", "")})
#     except Exception:
#         pass

#     # schedule LLM agents
#     llm_tasks = []
#     for agent_name in LLM_AGENTS_TO_RUN:
#         try:
#             chain = create_analysis_agent(agent_name)
#             llm_tasks.append((agent_name, chain.ainvoke({"transcript": transcript})))
#         except Exception:
#             all_scores.append({"agent_name": agent_name, "score": 0, "evidence": "Agent creation failed."})

#     llm_results = await asyncio.gather(*(t for _, t in llm_tasks), return_exceptions=True)
#     for i, result in enumerate(llm_results):
#         agent_name = llm_tasks[i][0]
#         if isinstance(result, Exception):
#             all_scores.append({"agent_name": agent_name, "score": 0, "evidence": "Agent execution failed."})
#         elif isinstance(result, dict):
#             all_scores.append({
#                 "agent_name": agent_name,
#                 "score": result.get("score", 0),
#                 "evidence": result.get("evidence", "No evidence provided.")
#             })
#         else:
#             all_scores.append({"agent_name": agent_name, "score": 0, "evidence": "Agent returned no output."})

#     return all_scores












##morninhg 4 
# backend/agent_manager.py
from __future__ import annotations
from typing import List, Union
import re

from backend.models import (
    AnalysisResult,
    AggregateAnalysis,
    AnalysisRequest,
    AnalysisResponse,
)

__all__ = [
    "run_all_agents",
    "run_analysis_pipeline",
    "results_to_llm_factor_rows",
    "AnalysisResult",
    "AggregateAnalysis",
]

# -------------------------
# Helpers: lenient scoring & richer evidence
# -------------------------

def _split_sentences(text: str) -> list[str]:
    # Simple sentence splitter; robust enough for our needs
    parts = re.split(r'(?<=[\.\!\?])\s+', (text or "").strip())
    return [p.strip() for p in parts if p and not p.isspace()]

def _find_hits(text: str, patterns: list[str]) -> list[tuple[int, int, str]]:
    """Return list of (start, end, match_text) for word-boundary matches."""
    hits = []
    if not text:
        return hits
    for p in patterns:
        for m in re.finditer(rf'\b{re.escape(p)}\b', text, flags=re.IGNORECASE):
            hits.append((m.start(), m.end(), m.group(0)))
    return hits

def _select_informative_sentences(sents: list[str], k: int = 2) -> list[str]:
    """
    Fallback when keyword hits are sparse:
    pick 1–2 moderately long, content-bearing sentences as evidence.
    """
    if not sents:
        return []
    # Preference: 40–220 chars; else take the longest ones
    candidates = [s for s in sents if 40 <= len(s) <= 220]
    if not candidates:
        candidates = sorted(sents, key=len, reverse=True)
    return candidates[:k]

def _evidence_from_patterns(text: str, patterns: list[str], max_items: int = 3) -> list[str]:
    """
    Pull up to `max_items` short evidence snippets (prefer whole sentences containing hits).
    If no keyword hits, fall back to informative sentences.
    """
    text = text or ""
    sents = _split_sentences(text)
    hits = _find_hits(text, patterns)

    if hits:
        evid = []
        for s in sents:
            if any(re.search(rf'\b{re.escape(p)}\b', s, flags=re.IGNORECASE) for p in patterns):
                evid.append(s.strip())
                if len(evid) >= max_items:
                    break
        # If somehow none caught, fall back to short local spans around the first few hits
        if not evid:
            for (start, end, _) in hits[:max_items]:
                span = text[max(0, start - 60): min(len(text), end + 60)].strip()
                evid.append(span)
        return evid[:max_items]

    # No hits: take 1–2 informative sentences as soft evidence
    return _select_informative_sentences(sents, k=min(2, max_items))

def _lenient_positive_score(hits_count: int, baseline: float = 0.65, step: float = 0.08, cap: float = 0.98) -> float:
    """Lenient mapping: even with no hits, give a neutral-positive score; evidence increases it."""
    if hits_count <= 0:
        return baseline
    return min(cap, baseline + hits_count * step)

def _lenient_negative_inverted(hits_count: int, start_hi: float = 0.88, step_down: float = 0.12, floor: float = 0.4) -> float:
    """For negative cues (brag/blame/know-it-all): start high and subtract per hit."""
    return max(floor, min(0.98, start_hi - hits_count * step_down))

def _agent_result(name: str, score_0_1: float, evid_list: list[str], extra: dict | None = None) -> AnalysisResult:
    label = "High" if score_0_1 >= 0.75 else ("Medium" if score_0_1 >= 0.55 else "Low")
    # Put evidence first in reasons (UI/Report uses this directly)
    reasons = [f'Evidence: “{e}”' for e in (evid_list or [])] or ["No explicit cues; lenient neutral scoring."]
    return AnalysisResult(agent=name, score=score_0_1, label=label, reasons=reasons, metrics=extra or {})

# -------------------------
# Factor calculators (lenient + rich evidence)
# -------------------------

def _admit_mistake(text: str) -> AnalysisResult:
    pats = ["sorry", "apologize", "apologies", "my fault", "i was wrong", "i made a mistake"]
    evid = _evidence_from_patterns(text, pats)
    hits = len(_find_hits(text, pats))
    score = _lenient_positive_score(hits, baseline=0.68, step=0.10)
    return _agent_result("AdmitMistake", score, evid, {"hits": hits})

def _mind_change(text: str) -> AnalysisResult:
    pats = ["i changed my mind", "i reconsidered", "i revised", "i updated my view", "i learned and changed", "pivoted"]
    evid = _evidence_from_patterns(text, pats)
    hits = len(_find_hits(text, pats))
    score = _lenient_positive_score(hits, baseline=0.63, step=0.11)
    return _agent_result("MindChange", score, evid, {"hits": hits})

def _share_credit(text: str) -> AnalysisResult:
    pats = ["we", "our team", "credit to", "thanks to", "with help from", "collaborated", "teammate", "pairing"]
    evid = _evidence_from_patterns(text, pats)
    hits = len(_find_hits(text, pats))
    score = _lenient_positive_score(hits, baseline=0.64, step=0.07)
    return _agent_result("ShareCredit", score, evid, {"hits": hits})

def _learner_mindset(text: str) -> AnalysisResult:
    pats = ["learn", "learned", "learning", "improve", "improved", "upskill", "curious", "feedback", "retrospective", "read docs", "mentor"]
    evid = _evidence_from_patterns(text, pats)
    hits = len(_find_hits(text, pats))
    score = _lenient_positive_score(hits, baseline=0.66, step=0.08)
    return _agent_result("LearnerMindset", score, evid, {"hits": hits})

def _brag_flag(text: str) -> AnalysisResult:
    pats = ["i alone", "only i", "single-handedly", "i was the best", "i outperformed everyone"]
    evid = _evidence_from_patterns(text, pats)
    hits = len(_find_hits(text, pats))
    score = _lenient_negative_inverted(hits, start_hi=0.90, step_down=0.15, floor=0.4)
    return _agent_result("BragFlag", score, evid, {"hits": hits})

def _blame_shift(text: str) -> AnalysisResult:
    pats = ["their fault", "they messed up", "not my fault", "because of them", "they didn't", "they failed"]
    evid = _evidence_from_patterns(text, pats)
    hits = len(_find_hits(text, pats))
    score = _lenient_negative_inverted(hits, start_hi=0.88, step_down=0.14, floor=0.4)
    return _agent_result("BlameShift", score, evid, {"hits": hits})

def _know_it_all(text: str) -> AnalysisResult:
    pats = ["i already knew", "obvious to me", "i never need help", "i know everything", "i don't need advice"]
    evid = _evidence_from_patterns(text, pats)
    hits = len(_find_hits(text, pats))
    score = _lenient_negative_inverted(hits, start_hi=0.86, step_down=0.13, floor=0.4)
    return _agent_result("KnowItAll", score, evid, {"hits": hits})

def _feedback_acceptance(text: str) -> AnalysisResult:
    pats = ["took feedback", "accepted feedback", "incorporated feedback", "acted on feedback", "mentor advice", "peer review", "retrospective action"]
    evid = _evidence_from_patterns(text, pats)
    hits = len(_find_hits(text, pats))
    score = _lenient_positive_score(hits, baseline=0.67, step=0.09)
    return _agent_result("FeedbackAcceptance", score, evid, {"hits": hits})

def _support_growth(text: str) -> AnalysisResult:
    pats = ["helped", "mentored", "coached", "supported", "unblocked", "enabled others", "guided"]
    evid = _evidence_from_patterns(text, pats)
    hits = len(_find_hits(text, pats))
    score = _lenient_positive_score(hits, baseline=0.64, step=0.09)
    return _agent_result("SupportGrowth", score, evid, {"hits": hits})

def _praise_handling(text: str) -> AnalysisResult:
    pats_pos = ["shared credit", "team effort", "grateful", "appreciate the team", "thanks to the team"]
    pats_neg = ["i deserved all the credit", "only i did this"]
    evid = _evidence_from_patterns(text, pats_pos + pats_neg)
    pos_hits = len(_find_hits(text, pats_pos))
    neg_hits = len(_find_hits(text, pats_neg))
    score = _lenient_positive_score(pos_hits, baseline=0.62, step=0.10)
    if neg_hits:
        score = max(0.4, score - 0.12 * neg_hits)
    return _agent_result("PraiseHandling", score, evid, {"pos_hits": pos_hits, "neg_hits": neg_hits})

# -------------------------
# Public API (sync)
# -------------------------

def run_all_agents(text: str) -> List[AnalysisResult]:
    text = (text or "").strip()
    low = text.lower()
    return [
        _admit_mistake(low),
        _mind_change(low),
        _share_credit(low),
        _learner_mindset(low),
        _brag_flag(low),
        _blame_shift(low),
        _know_it_all(low),
        _feedback_acceptance(low),
        _support_growth(low),
        _praise_handling(low),
    ]

def run_analysis_pipeline(request: Union[AnalysisRequest, str]) -> AnalysisResponse:
    if isinstance(request, str):
        req = AnalysisRequest(
            candidate_id="UNKNOWN",
            role="UNKNOWN",
            question="",
            answer=request,
            metadata={},
        )
    else:
        req = request

    results = run_all_agents(req.answer)

    aggregate = None
    if results:
        norm = [r.score if r.score <= 1.0 else r.score / 100.0 for r in results]
        overall = sum(norm) / len(norm)
        summary = " | ".join(
            f"{r.agent}: {r.label} ({(r.score*100 if r.score<=1 else r.score):.1f}%)"
            for r in results
        )
        aggregate = AggregateAnalysis(results=results, summary=summary, overall_score=overall)

    return AnalysisResponse(results=results, aggregate=aggregate, errors=[])

def results_to_llm_factor_rows(results: List[AnalysisResult]) -> list[dict]:
    """
    Rows like:
      {"agent_name": "AdmitMistake", "score": 7.2, "evidence": "• Evidence: “…” • Evidence: “…”"}
    (score on 0..10)
    """
    rows = []
    for r in results or []:
        score10 = (r.score * 10.0) if r.score <= 1.0 else (r.score / 10.0)
        # Join evidence bullets (reasons already hold “Evidence: …” lines)
        ev = " • ".join(r.reasons[:3]) if r.reasons else ""
        rows.append({"agent_name": r.agent, "score": float(score10), "evidence": ev})
    return rows
