# """
# Simple agent manager for humility analysis
# """

# import asyncio
# from typing import List, Dict, Any, NamedTuple

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

# def calculate_overall_score(results: Dict[str, tuple[float, str]]) -> float:
#     """Calculate an overall score from all analysis results"""
#     total = 0.0
#     count = 0
    
#     for score, _ in results.values():
#         total += score
#         count += 1
    
#     return round(total / count, 1) if count > 0 else 0.0

# async def run_analysis_pipeline(transcript: str) -> AnalysisResult:
#     """
#     Run all analysis agents on the transcript and return combined results.
    
#     Args:
#         transcript: The text to analyze
        
#     Returns:
#         AnalysisResult: Named tuple containing scores and evidence from all agents
#     """
#     # Run all analysis functions concurrently
#     results = await asyncio.gather(
#         asyncio.to_thread(analyze_humility, transcript),
#         asyncio.to_thread(analyze_learning, transcript),
#         asyncio.to_thread(analyze_feedback_seeking, transcript),
#         asyncio.to_thread(analyze_mistakes, transcript)
#     )
    
#     # Create a dictionary of results
#     result_dict = {
#         'humility': results[0],
#         'learning': results[1],
#         'feedback': results[2],
#         'mistakes': results[3]
#     }
    
#     # Calculate overall score
#     overall_score = calculate_overall_score(result_dict)
    
#     # Return results as a named tuple
#     return AnalysisResult(
#         humility_score=results[0][0],
#         humility_evidence=results[0][1],
#         learning_score=results[1][0],
#         learning_evidence=results[1][1],
#         feedback_score=results[2][0],
#         feedback_evidence=results[2][1],
#         mistakes_score=results[3][0],
#         mistakes_evidence=results[3][1],
#     )

# from .agents.humility_agent import analyze_humility
# from .agents.learning_agent import analyze_learning
# from .agents.feedback_agent import analyze_feedback_seeking
# from .agents.mistake_agent import analyze_mistakes






















"""
Simple agent manager for humility analysis
"""

import asyncio
from typing import List, Dict, Any, NamedTuple, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisResult(NamedTuple):
    """Container for analysis results from all agents"""
    humility_score: float
    humility_evidence: str
    learning_score: float
    learning_evidence: str
    feedback_score: float
    feedback_evidence: str
    mistakes_score: float
    mistakes_evidence: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the analysis result to a dictionary"""
        return {
            'humility': {'score': self.humility_score, 'evidence': self.humility_evidence},
            'learning': {'score': self.learning_score, 'evidence': self.learning_evidence},
            'feedback': {'score': self.feedback_score, 'evidence': self.feedback_evidence},
            'mistakes': {'score': self.mistakes_score, 'evidence': self.mistakes_evidence}
        }

def calculate_overall_score(results: Dict[str, Tuple[float, str]]) -> float:
    """Calculate an overall score from all analysis results"""
    total = 0.0
    count = 0
    
    for score, _ in results.values():
        total += score
        count += 1
    
    return total / count if count > 0 else 0.0

async def analyze_humility(text: str) -> Tuple[float, str]:
    """Analyze text for humility indicators"""
    if not text.strip():
        return 0.0, "No text provided for analysis"
    
    # Simple keyword-based analysis
    humility_indicators = [
        "i think", "maybe", "perhaps", "in my opinion", 
        "i believe", "i feel", "i would say", "i'm not sure",
        "i could be wrong", "correct me if i'm wrong"
    ]
    
    score = 0.0
    evidence = []
    
    for indicator in humility_indicators:
        if indicator in text.lower():
            score += 1.0
            evidence.append(indicator)
    
    # Cap score at 10
    score = min(10.0, score * 2)  # Scale to 0-10 range
    
    if evidence:
        return score, f"Found humility indicators: {', '.join(evidence)}"
    return score, "No strong indicators of humility found"

async def analyze_learning_mindset(text: str) -> Tuple[float, str]:
    """Analyze text for learning mindset indicators"""
    if not text.strip():
        return 0.0, "No text provided for analysis"
    
    learning_indicators = [
        "learned", "discovered", "realized", "understood",
        "grew", "improved", "developed", "gained insight",
        "expanded my knowledge", "new perspective"
    ]
    
    score = 0.0
    evidence = []
    
    for indicator in learning_indicators:
        if indicator in text.lower():
            score += 1.0
            evidence.append(indicator)
    
    # Cap score at 10
    score = min(10.0, score * 2)  # Scale to 0-10 range
    
    if evidence:
        return score, f"Found learning indicators: {', '.join(evidence)}"
    return score, "No strong indicators of learning mindset found"

async def analyze_feedback_seeking(text: str) -> Tuple[float, str]:
    """Analyze text for feedback seeking behavior"""
    if not text.strip():
        return 0.0, "No text provided for analysis"
    
    feedback_indicators = [
        "feedback", "suggestions", "advice", "input",
        "thoughts", "opinion", "what do you think",
        "how can i improve", "constructive criticism"
    ]
    
    score = 0.0
    evidence = []
    
    for indicator in feedback_indicators:
        if indicator in text.lower():
            score += 1.0
            evidence.append(indicator)
    
    # Cap score at 10
    score = min(10.0, score * 2)  # Scale to 0-10 range
    
    if evidence:
        return score, f"Found feedback seeking indicators: {', '.join(evidence)}"
    return score, "No strong indicators of feedback seeking found"

async def analyze_mistake_handling(text: str) -> Tuple[float, str]:
    """Analyze text for how mistakes are handled"""
    if not text.strip():
        return 0.0, "No text provided for analysis"
    
    positive_indicators = [
        "learned from", "grew from", "improved after",
        "took responsibility", "admitted", "acknowledged"
    ]
    
    negative_indicators = [
        "blamed", "excuses", "not my fault", "they made me",
        "had to", "no choice", "everyone else"
    ]
    
    score = 5.0  # Start with neutral score
    evidence = []
    
    # Check for positive indicators
    for indicator in positive_indicators:
        if indicator in text.lower():
            score += 1.0
            evidence.append(f"positive: {indicator}")
    
    # Check for negative indicators
    for indicator in negative_indicators:
        if indicator in text.lower():
            score -= 1.5
            evidence.append(f"negative: {indicator}")
    
    # Ensure score is between 0 and 10
    score = max(0.0, min(10.0, score * 2))  # Scale to 0-10 range
    
    if evidence:
        return score, f"Mistake handling indicators: {', '.join(evidence)}"
    return score, "No clear indicators of mistake handling found"

async def run_analysis_pipeline(text: str) -> AnalysisResult:
    """
    Run all analysis functions in parallel and return combined results
    
    Args:
        text: The text to analyze
        
    Returns:
        AnalysisResult: Combined results from all analysis functions
    """
    if not text.strip():
        return AnalysisResult(0.0, "No text", 0.0, "No text", 0.0, "No text", 0.0, "No text")
    
    try:
        # Run all analysis functions in parallel
        humility_score, humility_evidence = await analyze_humility(text)
        learning_score, learning_evidence = await analyze_learning_mindset(text)
        feedback_score, feedback_evidence = await analyze_feedback_seeking(text)
        mistakes_score, mistakes_evidence = await analyze_mistake_handling(text)
        
        return AnalysisResult(
            humility_score=humility_score,
            humility_evidence=humility_evidence,
            learning_score=learning_score,
            learning_evidence=learning_evidence,
            feedback_score=feedback_score,
            feedback_evidence=feedback_evidence,
            mistakes_score=mistakes_score,
            mistakes_evidence=mistakes_evidence
        )
    except Exception as e:
        logger.error(f"Error in analysis pipeline: {str(e)}")
        return AnalysisResult(0.0, "Error", 0.0, "Error", 0.0, "Error", 0.0, "Error")

# For backward compatibility
async def analyze_learning(*args, **kwargs):
    """Alias for analyze_learning_mindset"""
    return await analyze_learning_mindset(*args, **kwargs)

async def analyze_mistakes(*args, **kwargs):
    """Alias for analyze_mistake_handling"""
    return await analyze_mistake_handling(*args, **kwargs)