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























import sys
import os
from typing import List, NamedTuple , Tuple
import asyncio
from backend.agents.humility_agent import analyze_humility 
from backend.agents.learning_agent import analyze_learning as analyze_learning_mindset
from backend.agents.feedback_agent import analyze_feedback_seeking
from backend.agents.mistake_agent import analyze_mistake_handling
# Add the project root to the Python path
# This allows us to import modules from 'core' and 'services'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents import (
    create_analysis_agent,
    pronoun_ratio_agent,
    i_dont_know_agent
)
from core.state import AgentScore
class AnalysisResult(NamedTuple):          
    """Container for the four agent outputs."""                                 
    humility_score: float                  
    humility_evidence: str                 
    learning_score: float                  
    learning_evidence: str                 
    feedback_score: float                  
    feedback_evidence: str                 
    mistakes_score: float                  
    mistakes_evidence: str 
# These are the LLM-based agents that will be run on each transcript.
# You can easily add or remove agents from this list to change the analysis.
LLM_AGENTS_TO_RUN = [
    "AdmitMistakeAgent",
    "MindChangeAgent",
    "ShareCreditAgent",
    "LearnerMindsetAgent",
    "BragFlagAgent",
    "BlameShiftAgent",
    "KnowItAllAgent",
    "FeedbackAcceptanceAgent",
    "SupportGrowthAgent",
    "PraiseHandlingAgent" # This would be triggered conditionally in a more advanced setup
]

async def run_analysis_pipeline(transcript: str) -> List[AgentScore]:
    """
    Runs all relevant analysis agents on a given transcript.
    It runs simple parser-based agents and complex LLM-based agents in parallel.
    
    Args:
        transcript: The text of the candidate's response.

    Returns:
        A list of AgentScore objects from all agents.
    """
    all_scores: List[AgentScore] = []
    print("--- Starting Analysis Pipeline ---")

    # --- 1. Run Simple Parser Agents (Fast, No LLM calls) ---
    # These are synchronous and fast, so we run them directly.
    try:
        all_scores.append(pronoun_ratio_agent(transcript))
        all_scores.append(i_dont_know_agent(transcript))
        print("Executed simple parser agents.")
    except Exception as e:
        print(f"Error in simple parser agents: {e}")
        
    # --- 2. Run LLM-based Agents in Parallel ---
    llm_tasks = []
    for agent_name in LLM_AGENTS_TO_RUN:
        try:
            # Create the specific agent chain from the factory
            agent_chain = create_analysis_agent(agent_name)
            # Create an async task for each agent invocation. This schedules them to run concurrently.
            task = agent_chain.ainvoke({"transcript": transcript})
            llm_tasks.append((agent_name, task))
        except Exception as e:
            print(f"Error creating agent {agent_name}: {e}")

    print(f"Scheduled {len(llm_tasks)} LLM agents to run in parallel.")

    # Await all scheduled LLM agent tasks. `asyncio.gather` runs them concurrently.
    # `return_exceptions=True` prevents one failed agent from stopping all others.
    llm_results = await asyncio.gather(*(task for _, task in llm_tasks), return_exceptions=True)

    for i, result in enumerate(llm_results):
        agent_name = llm_tasks[i][0]
        if isinstance(result, Exception):
            all_scores.append({"agent_name": agent_name, "score": 0, "evidence": "Agent execution failed."})
        elif result and isinstance(result, dict):
            all_scores.append({
                "agent_name": agent_name,
                "score": result.get('score', 0),
                "evidence": result.get('evidence', 'No evidence provided.')
            })
        else:
            all_scores.append({"agent_name": agent_name, "score": 0, "evidence": "Agent returned no output."})

    return all_scores



