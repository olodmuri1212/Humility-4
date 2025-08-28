# # core/state.py

# from pydantic import BaseModel, Field
# from typing import List, Dict, Any
# import uuid

# class AgentScore(BaseModel):
#     """
#     A structured object to hold the output from a single analysis agent.
#     This ensures consistency in how scores are stored and processed.
#     """
#     agent_name: str
#     score: int = 0
#     evidence: str = "No evidence provided."

# class Turn(BaseModel):
#     """
#     Represents one complete turn in the conversation:
#     the interviewer's question, the candidate's transcribed answer,
#     and the list of all analyses performed on that answer.
#     """
#     question: str
#     transcript: str
#     analysis_results: List[AgentScore] = Field(default_factory=list)
#     # You can add more simple, non-LLM analysis here if needed
#     # e.g., we_vs_i_ratio: float = 0.0

# class InterviewState(BaseModel):
#     """
#     The main state object for the entire application. An instance of this
#     will be stored in Streamlit's session_state to persist data across reruns.
#     """
#     # A unique ID for each interview session.
#     interview_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
#     # Optional candidate name for the report
#     candidate_name: str | None = None

#     # The list of mandatory questions that must be asked.
#     # The list is consumed as the interview progresses.
#     mandatory_questions: List[str] = [
#         "Tell me about a time you were wrong.",
#         "When was the last time you changed your mind about something important?",
#         "Describe a success you're proud of. Who else deserves credit for it?",
#         "What’s something you’ve recently learned from someone junior to you?",
#     ]
    
#     # A log of all turns taken in the conversation.
#     conversation_history: List[Turn] = Field(default_factory=list)
    
#     # A dictionary to hold the running total score for each humility trait.
#     # e.g., {"AdmitMistakeAgent": 8, "BragFlagAgent": -5}
#     cumulative_scores: Dict[str, int] = Field(default_factory=dict)
    
#     # The overall humility score, normalized to a 0-100 scale.
#     # This is recalculated after each turn.
#     normalized_humility_score: int = 0
    
#     # The current status of the interview.
#     interview_status: str = "not_started" # Can be: not_started, in_progress, finished
















# from typing import Dict, List, Optional
# from pydantic import BaseModel, Field
# from typing_extensions import TypedDict

# class AgentScore(TypedDict):
#     """Score and evidence from a single analysis agent."""
#     agent_name: str
#     score: float
#     evidence: str

# class ConversationTurn(BaseModel):
#     """A single question-answer turn in the interview."""
#     question: str
#     transcript: str
#     analysis_results: List[AgentScore] = Field(default_factory=list)

# class InterviewState(BaseModel):
#     """Complete state of an interview session."""
#     candidate_name: str
#     normalized_humility_score: float = 0.0
#     cumulative_scores: Dict[str, float] = Field(default_factory=dict)
#     conversation_history: List[ConversationTurn] = Field(default_factory=list)

#     def to_dict(self) -> Dict:
#         """Convert the interview state to a dictionary for JSON serialization."""
#         return {
#             "candidate_name": self.candidate_name,
#             "normalized_humility_score": self.normalized_humility_score,
#             "cumulative_scores": self.cumulative_scores,
#             "conversation_history": [t.dict() for t in self.conversation_history],
#         }




















from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

class AgentScore(TypedDict):
    """Score and evidence from a single analysis agent."""
    agent_name: str
    score: float
    evidence: str

class ConversationTurn(BaseModel):
    """A single question-answer turn in the interview."""
    question: str
    transcript: str
    analysis_results: List[AgentScore] = Field(default_factory=list)

class InterviewState(BaseModel):
    """Complete state of an interview session."""
    candidate_name: str
    normalized_humility_score: float = 0.0
    cumulative_scores: Dict[str, float] = Field(default_factory=dict)
    conversation_history: List[ConversationTurn] = Field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert the interview state to a dictionary for JSON serialization."""
        return {
            "candidate_name": self.candidate_name,
            "normalized_humility_score": self.normalized_humility_score,
            "cumulative_scores": self.cumulative_scores,
            "conversation_history": [t.dict() for t in self.conversation_history],
        }
