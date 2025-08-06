"""
Pydantic models for the Humility Interview API
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel

class GenerateAudioRequest(BaseModel):
    text: str

class GenerateAudioResponse(BaseModel):
    audio_b64: str

class AnalyzeRequest(BaseModel):
    transcript: str
    question: Optional[str] = ""

class AnalysisResponse(BaseModel):
    scores: Dict[str, Dict[str, Any]]

class TranscribeRequest(BaseModel):
    audio_b64: str

class TranscribeResponse(BaseModel):
    transcript: str
