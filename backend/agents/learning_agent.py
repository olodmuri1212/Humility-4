"""Learning mindset analysis agent"""

def analyze_learning(transcript: str) -> tuple[float, str]:
    """
    Analyze learning mindset indicators in the transcript.
    Returns a tuple of (score, evidence)
    """
    learning_indicators = [
        "learn", "study", "research", "read", "explore", "discover",
        "understand", "figure out", "find out", "gain insight", "grow",
        "develop", "improve", "enhance", "progress", "advance", "master",
        "skill", "knowledge", "expertise", "competence", "ability"
    ]
    
    score = 0
    evidence = []
    
    # Check for learning indicators
    for indicator in learning_indicators:
        count = transcript.lower().count(indicator.lower())
        if count > 0:
            score += 0.5 * count
            evidence.append(f"Found learning indicator: '{indicator}'")
    
    # Check for past learning experiences
    learning_experience_phrases = [
        "i learned", "i discovered", "i found out", "i realized",
        "came to understand", "was able to learn", "expanded my knowledge"
    ]
    
    for phrase in learning_experience_phrases:
        if phrase in transcript.lower():
            score += 1.0
            evidence.append(f"Mentioned learning experience: '{phrase}'")
    
    # Cap the score at 10
    score = min(10.0, round(score, 1))
    
    if not evidence:
        evidence.append("No strong indicators of a learning mindset found in the response.")
    
    return score, "; ".join(evidence)
