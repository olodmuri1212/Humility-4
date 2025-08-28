# # interview_state.py
# from dataclasses import dataclass, field
# from typing import List, Dict, Union
# from backend.agent_manager import AnalysisResult

# @dataclass
# class Turn:
#     """Represents a single turn in the interview process."""
#     question: str
#     audio_data: tuple  # (sample_rate, np.ndarray)
#     transcript: str = ""
#     analysis_results: List[Union[AnalysisResult, tuple]] = field(default_factory=list)

# @dataclass
# class SessionState:
#     candidate_name: str = ""
#     turns: List[Turn] = field(default_factory=list)
#     cumulative_scores: Dict[str, float] = field(default_factory=dict)
#     normalized_humility_score: float = 0.0

#     def reset(self):
#         """Clear previous session data."""
#         self.candidate_name = ""
#         self.turns.clear()
#         self.cumulative_scores.clear()
#         self.normalized_humility_score = 0.0

#     def compute_session_scores(self):
#         """Sum agent scores over all turns and normalize to 0–100.
#         Handles both AnalysisResult objects and simple (score,evidence) tuples.
#         """
#         sums: Dict[str, float] = {}
#         n = len(self.turns) or 1

#         for turn in self.turns:
#             for res in turn.analysis_results:
#                 # Preferred: AnalysisResult
#                 if hasattr(res, "agent_name") and hasattr(res, "score"):
#                     name = res.agent_name
#                     score = res.score
#                 # Fallback: a tuple like (score, evidence)
#                 elif isinstance(res, (tuple, list)) and len(res) >= 1 and isinstance(res[0], (int, float)):
#                     name = "unknown_agent"
#                     score = res[0]
#                 else:
#                     continue

#                 sums[name] = sums.get(name, 0.0) + score

#         self.cumulative_scores = sums

#         # compute average per agent
#         avgs = [total / n for total in sums.values()] if sums else [0.0]
#         mean_avg = sum(avgs) / len(avgs)
#         # map 0–5 scale to 0–100
#         self.normalized_humility_score = round((mean_avg / 5.0) * 100, 1)

















# # interview_state.py

# from dataclasses import dataclass, field
# from typing import Any, Dict, List, Tuple

# @dataclass
# class Turn:
#     """One Q&A turn: question, raw audio, transcript, analysis results."""
#     question: str
#     audio_data: Tuple[int, Any]  # (sample_rate, np.ndarray)
#     transcript: str = ""
#     analysis_results: List[Dict] = field(default_factory=list)

# @dataclass
# class SessionState:
#     """Global interview session state."""
#     candidate_name: str = ""
#     turns: List[Turn] = field(default_factory=list)
#     cumulative_scores: Dict[str, float] = field(default_factory=dict)
#     normalized_humility_score: float = 0.0

#     def reset(self):
#         """Clear all prior data."""
#         self.candidate_name = ""
#         self.turns.clear()
#         self.cumulative_scores.clear()
#         self.normalized_humility_score = 0.0

#     def compute_session_scores(self):
#         """Sum agent scores over all turns and normalize to 0–100."""
#         sums: Dict[str, float] = {}
#         n = len(self.turns) or 1
#         for turn in self.turns:
#             for res in turn.analysis_results:
#                 name = res.get("agent_name")
#                 score = res.get("score", 0.0)
#                 if name:
#                     sums[name] = sums.get(name, 0.0) + score

#         self.cumulative_scores = sums
        # avgs = [total / n for total in sums.values()] if sums else [0.0]
        # mean_avg = sum(avgs) / len(avgs)
        # self.normalized_humility_score = round((mean_avg / 5.0) * 100, 1)

















##dopher 1##kaam kar raha hai
# interview_state.py (canonical)

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict

from backend.models import AnalysisResult, AggregateAnalysis
from backend.agent_manager import run_all_agents, results_to_llm_factor_rows

# The 10 factors considered for Humility Index (0–10)
TEN_FACTOR_SET = {
    "AdmitMistake",
    "MindChange",
    "ShareCredit",
    "LearnerMindset",
    "BragFlag",
    "BlameShift",
    "KnowItAll",
    "FeedbackAcceptance",
    "SupportGrowth",
    "PraiseHandling",
}

@dataclass
class Turn:
    """
    One Q&A interaction.
    - transcript/audio_data + legacy user_text
    - analyses: raw AnalysisResult list
    - aggregate: rollup (0..1 overall_score)
    - per_agent_rows: [{agent_name, score(0..10), evidence}]
    - humility_10 / learning_10 / feedback_10: per-question indices (0..10)
    """
    question: str
    audio_data: Optional[Any] = None
    transcript: str = ""
    user_text: str = ""
    ai_text: str = ""
    analyses: List[AnalysisResult] = field(default_factory=list)
    aggregate: Optional[AggregateAnalysis] = None

    per_agent_rows: List[Dict[str, Any]] = field(default_factory=list)
    humility_10: float = 0.0
    learning_10: float = 0.0
    feedback_10: float = 0.0

    def __post_init__(self) -> None:
        if not self.user_text and self.transcript:
            self.user_text = self.transcript

def _avg_or_zero(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 2) if vals else 0.0

def _summarize(analyses: List[AnalysisResult]) -> str:
    parts = [f"{a.agent}: {a.label} ({a.score_pct():.1f}%)" for a in analyses]
    return " | ".join(parts)

@dataclass
class SessionState:
    """
    Entire interview state for a single session.
    """
    candidate_id: str = "UNKNOWN"
    role: str = "UNKNOWN"
    turns: List[Turn] = field(default_factory=list)

    # ---- Lifecycle ----
    def reset(self, candidate_id: Optional[str] = None, role: Optional[str] = None) -> None:
        if candidate_id:
            self.candidate_id = candidate_id
        if role:
            self.role = role
        self.turns.clear()

    # ---- Mutators ----
    def set_candidate(self, candidate_id: str) -> None:
        if candidate_id:
            self.candidate_id = candidate_id

    def set_role(self, role: str) -> None:
        if role:
            self.role = role

    def start_turn(self, question: str) -> Turn:
        turn = Turn(question=question)
        self.turns.append(turn)
        return turn

    def add_user_answer(self, turn_index: int, answer: str) -> None:
        t = self._turn(turn_index)
        t.user_text = answer
        if not t.transcript:
            t.transcript = answer

    def attach_audio(self, turn_index: int, audio_data: Any) -> None:
        self._turn(turn_index).audio_data = audio_data

    def attach_transcript(self, turn_index: int, transcript: str) -> None:
        t = self._turn(turn_index)
        t.transcript = transcript
        if not t.user_text:
            t.user_text = transcript

    def analyze_turn(self, turn_index: int) -> List[AnalysisResult]:
        """
        Per-question analysis (distinct for each answer).
        Populates per_agent_rows + humility_10 / learning_10 / feedback_10.
        """
        turn = self._turn(turn_index)
        text = turn.user_text or turn.transcript or ""

        # 1) run agents
        analyses = run_all_agents(text)
        turn.analyses = analyses

        # 2) aggregate (0..1 overall)
        if analyses:
            norm = [a.score if a.score <= 1.0 else a.score / 100.0 for a in analyses]
            turn.aggregate = AggregateAnalysis(
                results=analyses,
                summary=_summarize(analyses),
                overall_score=sum(norm) / len(norm),
            )

        # 3) UI/report-friendly rows (0..10 + evidence)
        rows = results_to_llm_factor_rows(analyses)
        turn.per_agent_rows = rows

        # 4) indices
        row_map = {r["agent_name"]: float(r["score"]) for r in rows}
        # Humility: average of all ten
        turn.humility_10 = _avg_or_zero([row_map.get(n, 0.0) for n in TEN_FACTOR_SET])
        # Learning: LearnerMindset + KnowItAll
        turn.learning_10 = _avg_or_zero([
            row_map.get("LearnerMindset", 0.0),
            row_map.get("KnowItAll", 0.0),
        ])
        # Feedback: FeedbackAcceptance + PraiseHandling + ShareCredit
        turn.feedback_10 = _avg_or_zero([
            row_map.get("FeedbackAcceptance", 0.0),
            row_map.get("PraiseHandling", 0.0),
            row_map.get("ShareCredit", 0.0),
        ])

        return analyses

    def last_turn(self) -> Optional[Turn]:
        return self.turns[-1] if self.turns else None

    # ---- Internals ----
    def _turn(self, idx: int) -> Turn:
        if idx < 0 or idx >= len(self.turns):
            raise IndexError(f"Turn index out of range: {idx}")
        return self.turns[idx]
