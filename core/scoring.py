# core/scoring.py

# This dictionary defines the importance of each humility factor.
# The final score is a weighted average based on these values.
# The absolute sum of weights is 100, making normalization straightforward.
WEIGHTS = {
    # Core Question Agents (triggered by specific questions)
    "AdmitMistakeAgent": 8,
    "MindChangeAgent": 8,
    "ShareCreditAgent": 8,
    "LearnerMindsetAgent": 8,
    
    # Behavioral Agents (run on every response)
    "PronounRatioAgent": 6,
    "FeedbackAcceptanceAgent": 6,
    "SupportGrowthAgent": 6,
    "IDontKnowAgent": 6,
    "PraiseHandlingAgent": 6,
    
    # Penalty Agents (negative weights detract from the score)
    "BragFlagAgent": -10,
    "BlameShiftAgent": -8,
    "KnowItAllAgent": -6,

    # Agents not included in scoring for now but could be added
    "TalkListenAgent": 0,
    "InquiryAgent": 0,
    "InterruptionAgent": 0,
}

# The total possible positive score if a candidate scores a perfect 10 on all positive traits.
MAX_POSITIVE_SCORE_TOTAL = sum(w * 10 for w in WEIGHTS.values() if w > 0)


def calculate_normalized_score(cumulative_scores: dict) -> int:
    """
    Calculates a final score from 0-100 based on the weighted scores
    from all agents that have run so far.
    
    Args:
        cumulative_scores: A dictionary mapping agent names to their total raw scores.

    Returns:
        A final, normalized score between 0 and 100.
    """
    if not cumulative_scores:
        return 0

    raw_weighted_score = 0
    for agent_name, total_raw_score in cumulative_scores.items():
        weight = WEIGHTS.get(agent_name, 0)
        raw_weighted_score += total_raw_score * weight

    # If no positive-weighted agents have run, the score is 0.
    if MAX_POSITIVE_SCORE_TOTAL == 0:
        return 0
        
    # Normalize the score against the maximum possible positive score.
    # Penalties will naturally pull this percentage down.
    normalized_score = (raw_weighted_score / MAX_POSITIVE_SCORE_TOTAL) * 100
    
    # Clamp the final score to be within the 0-100 range.
    final_score = max(0, min(100, int(normalized_score)))
    
    return final_score 