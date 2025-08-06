"""Humility analysis agent"""
import asyncio

def analyze_humility(transcript: str) -> tuple[float, str]:
    """
    Analyze humility indicators in the transcript.
    Returns a tuple of (score, evidence)
    """
    humble_phrases = [
        "i could be wrong", "i might be mistaken", "in my opinion",
        "from my perspective", "i think", "i believe", "i feel",
        "i learned", "i grew", "i improved", "i understand",
        "i appreciate", "thank you", "grateful", "acknowledge",
        "recognize", "admit", "own up to", "take responsibility"
    ]
    
    humble_indicators = [
        "mistake", "learn", "grow", "improve", "feedback",
        "help", "support", "team", "collaborat", "together"
    ]
    
    score = 0
    evidence = []
    
    # Check for humble phrases
    for phrase in humble_phrases:
        if phrase in transcript.lower():
            score += 0.5
            evidence.append(f"Found humble phrase: '{phrase}'")
    
    # Check for humble indicators
    for indicator in humble_indicators:
        count = transcript.lower().count(indicator.lower())
        if count > 0:
            score += 0.3 * count
            evidence.append(f"Found {count} instance(s) of '{indicator}'")
    
    # Cap the score at 10
    score = min(10.0, round(score, 1))
    
    # If no evidence found, provide a default message
    if not evidence:
        evidence.append("No strong indicators of humility found in the response.")
    
    return score, "; ".join(evidence)
