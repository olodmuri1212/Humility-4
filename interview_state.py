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

















# interview_state.py

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

@dataclass
class Turn:
    """One Q&A turn: question, raw audio, transcript, analysis results."""
    question: str
    audio_data: Tuple[int, Any]  # (sample_rate, np.ndarray)
    transcript: str = ""
    analysis_results: List[Dict] = field(default_factory=list)

@dataclass
class SessionState:
    """Global interview session state."""
    candidate_name: str = ""
    turns: List[Turn] = field(default_factory=list)
    cumulative_scores: Dict[str, float] = field(default_factory=dict)
    normalized_humility_score: float = 0.0

    def reset(self):
        """Clear all prior data."""
        self.candidate_name = ""
        self.turns.clear()
        self.cumulative_scores.clear()
        self.normalized_humility_score = 0.0

    def compute_session_scores(self):
        """Sum agent scores over all turns and normalize to 0–100."""
        sums: Dict[str, float] = {}
        n = len(self.turns) or 1
        for turn in self.turns:
            for res in turn.analysis_results:
                name = res.get("agent_name")
                score = res.get("score", 0.0)
                if name:
                    sums[name] = sums.get(name, 0.0) + score

        self.cumulative_scores = sums
        avgs = [total / n for total in sums.values()] if sums else [0.0]
        mean_avg = sum(avgs) / len(avgs)
        self.normalized_humility_score = round((mean_avg / 5.0) * 100, 1)
