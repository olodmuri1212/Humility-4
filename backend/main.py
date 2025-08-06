# # backend/main.py

# import sys
# import os

# # Add the project root to the Python path
# # This allows us to import modules from 'core' and 'services'
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# import base64
# import tempfile

# # Import the Pydantic models for request/response validation
# from .models import (
#     TranscribeRequest, 
#     AnalysisResponse, 
#     AnalyzeRequest,
#     GenerateAudioRequest,
#     GenerateAudioResponse
# )
# # Import the core logic for running the agent pipeline
# from .agent_manager import run_analysis_pipeline

# # Import service functions that interact with third-party APIs
# from services.audio_processing import get_transcript_from_audio, get_speech_from_text

# # Initialize FastAPI app
# app = FastAPI(
#     title="Humility Interview Backend",
#     description="Provides endpoints for transcription, analysis, and text-to-speech.",
#     version="1.0.0"
# )

# # Configure CORS (Cross-Origin Resource Sharing)
# # This allows the Streamlit frontend (on a different port) to communicate with this backend.
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # In production, restrict this to your Streamlit app's domain
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
#     allow_headers=["*"],  # Allows all headers
# )

# @app.get("/", tags=["Status"])
# def read_root():
#     """A simple health check endpoint to confirm the server is running."""
#     return {"status": "Humility Interview Backend is online."}

# @app.get("/health", tags=["Status"])
# def health_check():
#     """Health check endpoint for testing."""
#     return {"status": "OK", "message": "Backend is healthy"}


# @app.post("/transcribe", response_model=dict, tags=["Audio Processing"])
# async def transcribe_audio(request: TranscribeRequest):
#     """
#     Receives base64 encoded audio, saves it to a temporary file, 
#     transcribes it using a remote service, and returns the text.
#     """
#     try:
#         # Decode the base64 string to get the raw audio bytes
#         audio_bytes = base64.b64decode(request.audio_b64)
        
#         # Using a temporary file is often more reliable for API uploads than in-memory buffers
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
#             tmp_audio_file.write(audio_bytes)
#             tmp_audio_path = tmp_audio_file.name

#         # Call the service function to get the transcript
#         transcript = get_transcript_from_audio(tmp_audio_path)
        
#     except Exception as e:
#         print(f"Error during transcription process: {e}")
#         raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")
#     finally:
#         # Ensure the temporary file is always deleted
#         if 'tmp_audio_path' in locals() and os.path.exists(tmp_audio_path):
#             os.unlink(tmp_audio_path)

#     if transcript is None:
#         raise HTTPException(status_code=500, detail="Failed to get a valid transcript from the audio service.")
            
#     return {"transcript": transcript}


# @app.post("/analyze", response_model=AnalysisResponse, tags=["AI Analysis"])
# async def analyze_transcript(request: AnalyzeRequest):
#     """
#     Receives a transcript and runs the full humility analysis pipeline on it.
#     """
#     if not request.transcript:
#         raise HTTPException(status_code=400, detail="Transcript cannot be empty.")

#     try:
#         # This is the core call to the agent manager
#         scores_list = await run_analysis_pipeline(request.transcript)
        
#         # Convert the list of Pydantic models to a dictionary for the JSON response
#         scores_dict = {
#             score.agent_name: {"score": score.score, "evidence": score.evidence} 
#             for score in scores_list
#         }

#         return AnalysisResponse(scores=scores_dict)
#     except Exception as e:
#         print(f"Error during analysis pipeline: {e}")
#         # Provide a generic error to the client for security
#         raise HTTPException(status_code=500, detail="An internal error occurred during analysis.")


# @app.post("/generate_audio", response_model=GenerateAudioResponse, tags=["Audio Processing"])
# async def generate_question_audio(request: GenerateAudioRequest):
#     """
#     Receives text and generates the audio for the interviewer's question using a TTS service.
#     """
#     if not request.text:
#         raise HTTPException(status_code=400, detail="Text for audio generation is required.")
    
#     try:
#         # Call the service function to get the audio bytes
#         audio_bytes = await get_speech_from_text(request.text)
#         if not audio_bytes:
#             raise HTTPException(status_code=500, detail="Failed to generate speech audio from the service.")
            
#         # Encode the audio bytes to base64 to send in a JSON response
#         audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
#         return GenerateAudioResponse(audio_b64=audio_b64)
        
#     except Exception as e:
#         print(f"Error during audio generation: {e}")
#         raise HTTPException(status_code=500, detail="An internal error occurred during audio generation.")

















# """
# FastAPI backend – now async end‑to‑end and Pipecat‑aware.
# """

# import base64, os, asyncio, tempfile, sys
# from pathlib import Path
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware

# # ---------------------------------------------------------------------------
# #  Make sure the top‑level package can be imported
# # ---------------------------------------------------------------------------
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.append(str(PROJECT_ROOT))

# from .models import (
#     TranscribeRequest,
#     AnalyzeRequest,
#     AnalysisResponse,
#     GenerateAudioRequest,
#     GenerateAudioResponse,
# )
# from .agent_manager import run_analysis_pipeline
# from services.audio_processing import get_transcript_from_audio, get_speech_from_text


# app = FastAPI(
#     title="Humility Interview Backend",
#     description="Transcription, analysis, and TTS endpoints powered by Pipecat.",
#     version="2.0.0",
# )

# # ---------------------------------------------------------------------------
# #  CORS – allow everything in dev, lock down in prod
# # ---------------------------------------------------------------------------
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# @app.get("/", tags=["Status"])
# def root():
#     return {"status": "online"}


# @app.get("/health", tags=["Status"])
# def health():
#     return {"status": "OK"}


# # ---------------------------------------------------------------------------
# #  /transcribe  ─ POST base64‑audio → text
# # ---------------------------------------------------------------------------
# @app.post("/transcribe", response_model=dict, tags=["Audio Processing"])
# async def transcribe_audio(req: TranscribeRequest):
#     if not req.audio_b64:
#         raise HTTPException(status_code=400, detail="Audio payload required")

#     audio_bytes = base64.b64decode(req.audio_b64)

#     # Whisper is CPU‑heavy – push to default thread pool
#     loop = asyncio.get_running_loop()
#     transcript = await loop.run_in_executor(
#         None, lambda: asyncio.run(get_transcript_from_audio(audio_bytes))
#     )

#     if not transcript:
#         raise HTTPException(
#             status_code=500, detail="Transcription failed – see server logs"
#         )
#     return {"transcript": transcript}


# # ---------------------------------------------------------------------------
# #  /analyze  ─ POST transcript → scores
# # ---------------------------------------------------------------------------
# @app.post("/analyze", response_model=AnalysisResponse, tags=["AI Analysis"])
# async def analyze_transcript(req: AnalyzeRequest):
#     if not req.transcript:
#         raise HTTPException(status_code=400, detail="Transcript cannot be empty")

#     try:
#         scores_list = await run_analysis_pipeline(req.transcript)
#         scores = {
#             s.agent_name: {"score": s.score, "evidence": s.evidence}
#             for s in scores_list
#         }
#         return AnalysisResponse(scores=scores)
#     except Exception as exc:
#         print("Pipeline error:", exc)
#         raise HTTPException(status_code=500, detail="Analysis pipeline failed")


# # ---------------------------------------------------------------------------
# #  /generate_audio  ─ POST text → base64‑audio
# # ---------------------------------------------------------------------------
# @app.post("/generate_audio", response_model=GenerateAudioResponse, tags=["Audio Processing"])
# async def generate_question_audio(req: GenerateAudioRequest):
#     if not req.text:
#         raise HTTPException(status_code=400, detail="Text required")

#     audio_bytes = await get_speech_from_text(req.text)
#     if not audio_bytes:
#         raise HTTPException(status_code=500, detail="TTS generation failed")

#     audio_b64 = base64.b64encode(audio_bytes).decode()
#     return GenerateAudioResponse(audio_b64=audio_b64)



















# backend/main.py
# """
# FastAPI app – transcription, analysis, TTS endpoints. :contentReference[oaicite:4]{index=4}
# """
# import sys, os, base64, asyncio
# from pathlib import Path
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware

# # allow imports from project root
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.append(str(PROJECT_ROOT))

# from .models import (
#     TranscribeRequest, AnalyzeRequest,
#     GenerateAudioRequest, GenerateAudioResponse,
#     AnalysisResponse
# )
# from .agent_manager import run_analysis_pipeline
# from services.audio_processing import (
#     get_transcript_from_audio, get_speech_from_text
# )

# app = FastAPI(
#     title="Humility Interview Backend",
#     version="2.0",
#     description="STT, analysis & TTS powered by Pipecat"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],    # lock this down in prod!
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/", tags=["Status"])
# def root():
#     return {"status": "online"}

# @app.post("/transcribe", response_model=dict, tags=["Audio Processing"])
# async def transcribe_audio(req: TranscribeRequest):
#     if not req.audio_b64:
#         raise HTTPException(400, "Audio payload required")
#     data = base64.b64decode(req.audio_b64)

#     # off‑load Whisper CPU work
#     loop = asyncio.get_running_loop()
#     transcript = await loop.run_in_executor(
#         None, lambda: asyncio.run(get_transcript_from_audio(data))
#     )
#     if not transcript:
#         raise HTTPException(500, "Transcription failed")
#     return {"transcript": transcript}

# @app.post("/analyze", response_model=AnalysisResponse, tags=["AI Analysis"])
# async def analyze(req: AnalyzeRequest):
#     if not req.transcript:
#         raise HTTPException(400, "Transcript cannot be empty")
#     try:
#         results = await run_analysis_pipeline(req.transcript)
#         scores = {
#             r.agent_name: {"score": r.score, "evidence": r.evidence}
#             for r in results
#         }
#         return AnalysisResponse(scores=scores)
#     except Exception as e:
#         print("Pipeline error:", e)
#         raise HTTPException(500, "Analysis pipeline failed")

# # @app.post("/generate_audio", response_model=GenerateAudioResponse, tags=["Audio Processing"])
# # async def generate_audio(req: GenerateAudioRequest):
# #     if not req.text:
# #         raise HTTPException(400, "Text is required")
# #     audio = await get_speech_from_text(req.text)
# #     if not audio:
# #         raise HTTPException(500, "TTS generation failed")
# #     return GenerateAudioResponse(audio_b64=base64.b64encode(audio).decode())


# @app.post("/generate_audio", response_model=GenerateAudioResponse)
# async def generate_audio(req: GenerateAudioRequest):
#     if not req.text:
#         raise HTTPException(400, "Text is required")
#     try:
#         audio = await get_speech_from_text(req.text)
#     except Exception as e:
#         print("💥 generate_audio fatal error:", e)
#         raise HTTPException(500, f"TTS error: {e}")
#     if not audio:
#         # now we know why from the logs above
#         raise HTTPException(500, "TTS generation failed – check server logs")
#     return GenerateAudioResponse(audio_b64=base64.b64encode(audio).decode())





























# # backend/main.py
# import sys, base64, asyncio
# from pathlib import Path
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware

# # allow imports from project root
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.append(str(PROJECT_ROOT))

# from .models import (
#     TranscribeRequest, AnalyzeRequest,
#     GenerateAudioRequest, GenerateAudioResponse,
#     AnalysisResponse
# )
# from .agent_manager import run_analysis_pipeline
# from services.audio_processing import (
#     get_transcript_from_audio,
#     get_speech_from_text
# )

# app = FastAPI(
#     title="Humility Interview Backend",
#     version="3.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # tighten in prod
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/", tags=["Status"])
# def root():
#     return {"status": "online"}

# @app.post("/transcribe", response_model=dict, tags=["Audio Processing"])
# async def transcribe_audio(req: TranscribeRequest):
#     if not req.audio_b64:
#         raise HTTPException(400, "audio_b64 is required")
#     audio_bytes = base64.b64decode(req.audio_b64)

#     loop = asyncio.get_running_loop()
#     transcript = await loop.run_in_executor(
#         None,
#         lambda: asyncio.run(get_transcript_from_audio(audio_bytes))
#     )
#     if not transcript:
#         raise HTTPException(500, "Transcription failed")
#     return {"transcript": transcript}

# @app.post("/analyze", response_model=AnalysisResponse, tags=["AI Analysis"])
# async def analyze(req: AnalyzeRequest):
#     if not req.transcript:
#         raise HTTPException(400, "Transcript cannot be empty")
#     try:
#         results = await run_analysis_pipeline(req.transcript)
#         return AnalysisResponse(
#             scores={r.agent_name: {"score": r.score, "evidence": r.evidence}
#                     for r in results}
#         )
#     except Exception as e:
#         print("Pipeline error:", e)
#         raise HTTPException(500, "Analysis pipeline failed")

# @app.post("/generate_audio", response_model=GenerateAudioResponse, tags=["Audio Processing"])
# async def generate_audio(req: GenerateAudioRequest):
#     if not req.text:
#         raise HTTPException(400, "Text is required")
#     audio_bytes = await get_speech_from_text(req.text)
#     if not audio_bytes:
#         raise HTTPException(500, "TTS generation failed – check server log")
#     return GenerateAudioResponse(audio_b64=base64.b64encode(audio_bytes).decode())



















# #backend.main.py
# import sys
# from pathlib import Path

# # Allow imports from project root
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.append(str(PROJECT_ROOT))

# import asyncio
# import base64

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware

# from .models import (
#     TranscribeRequest,
#     AnalyzeRequest,
#     GenerateAudioRequest,
#     GenerateAudioResponse,
#     AnalysisResponse,
# )
# from .agent_manager import run_analysis_pipeline
# from services.audio_processing import get_speech_from_text

# # Initialize FastAPI app
# app = FastAPI(
#     title="Humility Interview Backend",
#     version="3.1"
# )

# # Configure CORS (tighten origins in production)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/", tags=["Status"])
# def health_check():
#     """Simple health check endpoint."""
#     return {"status": "online"}

# @app.post(
#     "/generate_audio",
#     response_model=GenerateAudioResponse,
#     tags=["Audio Processing"]
# )
# async def generate_audio(req: GenerateAudioRequest):
#     """Generate TTS audio from input text."""
#     if not req.text:
#         raise HTTPException(status_code=400, detail="Text is required")

#     audio_bytes = await get_speech_from_text(req.text)
#     if not audio_bytes:
#         raise HTTPException(status_code=500, detail="TTS generation failed")

#     audio_b64 = base64.b64encode(audio_bytes).decode()
#     return GenerateAudioResponse(audio_b64=audio_b64)

# @app.post(
#     "/analyze",
#     response_model=AnalysisResponse,
#     tags=["AI Analysis"]
# )
# async def analyze(req: AnalyzeRequest):
#     """Run the humility analysis pipeline on a transcript."""
#     if not req.transcript:
#         raise HTTPException(status_code=400, detail="Transcript cannot be empty")

#     try:
#         results = await run_analysis_pipeline(req.transcript)
#         scores = {
#             r.agent_name: {"score": r.score, "evidence": r.evidence}
#             for r in results
#         }
#         return AnalysisResponse(scores=scores)
#     except Exception as e:
#         print("Analysis pipeline error:", e)
#         raise HTTPException(status_code=500, detail="Analysis pipeline failed")
































# import sys, base64, tempfile, subprocess, os
# from pathlib import Path
# from fastapi import FastAPI, HTTPException, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.append(str(PROJECT_ROOT))

# from .models import (AnalyzeRequest, AnalysisResponse,
#                      GenerateAudioRequest, GenerateAudioResponse)
# from .agent_manager import run_analysis_pipeline
# from services.audio_processing import (get_speech_from_text,
#                                        get_transcript_from_audio_sync)

# app = FastAPI(title="Humility Interview Backend", version="3.2")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], allow_credentials=True,
#     allow_methods=["*"], allow_headers=["*"],
# )

# @app.get("/", tags=["Status"])
# def health(): return {"status": "online"}

# # ────────────────── TTS ─────────────────────────────────────────────
# @app.post("/generate_audio", response_model=GenerateAudioResponse,
#           tags=["Audio Processing"])
# async def generate_audio(req: GenerateAudioRequest):
#     if not req.text:
#         raise HTTPException(400, "Text is required")
#     audio = await get_speech_from_text(req.text)
#     if not audio:
#         raise HTTPException(500, "TTS failed")
#     return {"audio_b64": base64.b64encode(audio).decode()}

# # ────────────────── Analysis ────────────────────────────────────────
# @app.post("/analyze", response_model=AnalysisResponse, tags=["AI Analysis"])
# async def analyze(req: AnalyzeRequest):
#     if not req.transcript:
#         raise HTTPException(400, "Transcript cannot be empty")
#     try:
#         res = await run_analysis_pipeline(req.transcript)
#         return {"scores": {r.agent_name: {"score": r.score,
#                                           "evidence": r.evidence}
#                            for r in res}}
#     except Exception as e:
#         print("Pipeline error:", e)
#         raise HTTPException(500, "Analysis failed")

# # ────────────────── STT fallback (from WAV/MP3) ─────────────────────
# def _maybe_to_wav(src: str) -> str:
#     """If src is MP3, convert to WAV @16 kHz mono via ffmpeg, else return src."""
#     if src.lower().endswith(".wav"):
#         return src
#     wav = f"{src}.wav"
#     subprocess.check_call([
#         "ffmpeg", "-y", "-i", src,
#         "-ar", "16000", "-ac", "1", wav
#     ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     return wav

# @app.post("/transcribe", response_model=dict, tags=["Audio Processing"])
# async def transcribe(file: UploadFile = File(...)):
#     tmp = tempfile.NamedTemporaryFile(delete=False,
#                                       suffix=Path(file.filename).suffix)
#     tmp.write(await file.read()); tmp.flush(); tmp.close()
#     wav_path = _maybe_to_wav(tmp.name)
#     text = get_transcript_from_audio_sync(Path(wav_path).read_bytes())
#     os.unlink(tmp.name);  # clean temp files
#     if wav_path != tmp.name: os.unlink(wav_path)
#     if not text:
#         raise HTTPException(500, "Transcription failed")
#     return {"transcript": text}






# from fastapi import WebSocket, WebSocketDisconnect
# from services.audio_processing import stream_transcript_from_chunks
# import asyncio, uuid, json, tempfile, wave, struct

# # ────────────────── Live STT (web‑socket) ──────────────────────────
# @app.websocket("/stt_stream")
# async def stt_stream(ws: WebSocket):
#     """
#     Bidirectional stream.
#     ↑ Client sends raw 16‑bit little‑endian mono 16 kHz PCM *bytes* every ~200 ms
#     ↓ Server sends JSON { "id": <chunk‑id>, "partial": <text>, "final": bool }
#     """
#     await ws.accept()
#     aggregator = stream_transcript_from_chunks()        # generator
#     try:
#         while True:
#             pcm_chunk = await ws.receive_bytes()         # <-- ~6400 bytes == 0.2 s
#             for update in aggregator.send(pcm_chunk):    # 0–many updates back
#                 await ws.send_text(json.dumps(update))
#     except WebSocketDisconnect:
#         aggregator.close()
#     except Exception as e:
#         await ws.close(code=1011, reason=str(e))





















##claud ai
#!/usr/bin/env python3
"""
Enhanced Humility Interview Backend
Improved audio processing and real-time STT support
"""

import sys
import base64
import tempfile
import subprocess
import os
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import models and services
try:
    from backend.models import (
        AnalyzeRequest, AnalysisResponse,
        GenerateAudioRequest, GenerateAudioResponse
    )
    from backend.agent_manager import run_analysis_pipeline
    from services.audio_processing import (
        get_speech_from_text,
        get_transcript_from_audio_sync,
        get_transcript_from_audio
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available.")
    sys.exit(1)

# Create FastAPI app
app = FastAPI(
    title="Humility Interview Backend",
    version="3.2",
    description="AI-Powered Interview Analysis with Real-time STT/TTS"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (for STT script)
try:
    app.mount("/static", StaticFiles(directory="public"), name="static")
except RuntimeError:
    # Create public directory if it doesn't exist
    os.makedirs("public", exist_ok=True)
    app.mount("/static", StaticFiles(directory="public"), name="static")

# Health check endpoint
@app.get("/", tags=["Status"])
def health_check():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Humility Interview Backend",
        "version": "3.2",
        "endpoints": {
            "tts": "/generate_audio",
            "stt": "/transcribe",
            "analysis": "/analyze",
            "health": "/"
        }
    }

# Text-to-Speech endpoint
@app.post("/generate_audio", response_model=GenerateAudioResponse, tags=["Audio Processing"])
async def generate_audio(req: GenerateAudioRequest):
    """Convert text to speech"""
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required and cannot be empty")
    
    try:
        print(f"TTS request: {req.text[:50]}...")
        audio_bytes = await get_speech_from_text(req.text.strip())
        
        if not audio_bytes:
            raise HTTPException(status_code=500, detail="Text-to-speech generation failed")
        
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        print(f"TTS success: Generated {len(audio_bytes)} bytes of audio")
        
        return GenerateAudioResponse(audio_b64=audio_b64)
        
    except Exception as e:
        print(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS processing failed: {str(e)}")

# Analysis endpoint
@app.post("/analyze", response_model=AnalysisResponse, tags=["AI Analysis"])
async def analyze_response(req: AnalyzeRequest):
    """Analyze transcript for humility indicators"""
    if not req.transcript or not req.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript cannot be empty")
    
    try:
        print(f"Analysis request: {req.transcript[:100]}...")
        results = await run_analysis_pipeline(req.transcript.strip())
        
        if not results:
            raise HTTPException(status_code=500, detail="Analysis returned no results")
        
        # Format results for response
        scores = {}
        for result in results:
            scores[result.agent_name] = {
                "score": result.score,
                "evidence": result.evidence
            }
        
        print(f"Analysis success: {len(scores)} agents analyzed")
        return AnalysisResponse(scores=scores)
        
    except Exception as e:
        print(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Enhanced STT endpoint for file uploads
@app.post("/transcribe", tags=["Audio Processing"])
async def transcribe_audio_file(file: UploadFile = File(...)):
    """Transcribe uploaded audio file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file type
    allowed_extensions = {'.wav', '.mp3', '.m4a', '.ogg', '.flac'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    temp_file = None
    wav_file = None
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            temp_input_path = temp_file.name
        
        print(f"STT file upload: {file.filename} ({len(content)} bytes)")
        
        # Convert to WAV if needed
        wav_path = convert_to_wav(temp_input_path)
        
        # Transcribe
        with open(wav_path, 'rb') as wav_file:
            audio_bytes = wav_file.read()
        
        transcript = get_transcript_from_audio_sync(audio_bytes)
        
        if not transcript:
            raise HTTPException(status_code=500, detail="Transcription failed - no speech detected")
        
        print(f"STT success: {transcript[:100]}...")
        
        return {"transcript": transcript.strip()}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"STT error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        # Cleanup temporary files
        try:
            if temp_input_path and os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if wav_path and wav_path != temp_input_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except:
            pass

# Real-time STT endpoint for base64 audio data
@app.post("/transcribe_realtime", tags=["Audio Processing"])
async def transcribe_realtime(data: Dict[str, Any]):
    """Transcribe base64-encoded audio data"""
    try:
        audio_b64 = data.get("audio_b64", "")
        if not audio_b64:
            raise HTTPException(status_code=400, detail="audio_b64 field is required")
        
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {e}")
        
        if len(audio_bytes) < 100:  # Minimum reasonable audio size
            raise HTTPException(status_code=400, detail="Audio data too short")
        
        print(f"Real-time STT: Processing {len(audio_bytes)} bytes")
        
        # Transcribe using async function
        transcript = await get_transcript_from_audio(audio_bytes)
        
        if not transcript:
            return {"transcript": "", "message": "No speech detected"}
        
        print(f"Real-time STT success: {transcript[:50]}...")
        return {"transcript": transcript.strip()}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Real-time STT error: {e}")
        raise HTTPException(status_code=500, detail=f"Real-time transcription failed: {str(e)}")

def convert_to_wav(input_path: str) -> str:
    """Convert audio file to WAV format using ffmpeg"""
    input_path = Path(input_path)
    
    if input_path.suffix.lower() == '.wav':
        return str(input_path)
    
    output_path = input_path.with_suffix('.wav')
    
    try:
        # Use ffmpeg to convert to 16kHz mono WAV
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",      # Mono
            "-c:a", "pcm_s16le",  # 16-bit PCM
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            raise Exception(f"ffmpeg failed: {result.stderr}")
        
        return str(output_path)
        
    except subprocess.TimeoutExpired:
        raise Exception("Audio conversion timed out")
    except FileNotFoundError:
        raise Exception("ffmpeg not found. Please install ffmpeg.")
    except Exception as e:
        raise Exception(f"Audio conversion failed: {e}")

# System info endpoint
@app.get("/system_info", tags=["Status"])
def get_system_info():
    """Get system information and service status"""
    info = {
        "backend_status": "online",
        "services": {
            "tts": "available",
            "stt": "available", 
            "analysis": "available"
        },
        "audio_formats": {
            "input": [".wav", ".mp3", ".m4a", ".ogg", ".flac"],
            "output": [".wav"]
        },
        "features": {
            "real_time_stt": True,
            "file_upload_stt": True,
            "multi_agent_analysis": True,
            "cors_enabled": True
        }
    }
    
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      capture_output=True, 
                      timeout=5)
        info["services"]["ffmpeg"] = "available"
    except:
        info["services"]["ffmpeg"] = "unavailable"
        info["warnings"] = ["ffmpeg not found - some audio formats may not work"]
    
    return info

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": str(asyncio.get_event_loop().time())
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500
        }
    )

# Development server
if __name__ == "__main__":
    print("🚀 Starting Humility Interview Backend...")
    print("📋 Available endpoints:")
    print("   - Health: GET /")
    print("   - TTS: POST /generate_audio")
    print("   - STT (file): POST /transcribe")
    print("   - STT (realtime): POST /transcribe_realtime")
    print("   - Analysis: POST /analyze")
    print("   - System Info: GET /system_info")
    
    uvicorn.run(
        "main:app",
        host="127.0.0.1",  # Changed from 0.0.0.0 to localhost for security
        port=8000,
        reload=True,
        log_level="info"
    )



















# import sys
# import base64
# import tempfile
# import subprocess
# import os
# import json
# from pathlib import Path

# from fastapi import (
#     FastAPI, HTTPException, UploadFile, File,
#     WebSocket, WebSocketDisconnect
# )
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles

# # Ensure project root is on sys.path for imports
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.append(str(PROJECT_ROOT))

# from .models import (
#     AnalyzeRequest, AnalysisResponse,
#     GenerateAudioRequest, GenerateAudioResponse
# )
# from .agent_manager import run_analysis_pipeline
# from services.audio_processing import (
#     get_speech_from_text,
#     get_transcript_from_audio_sync,
#     stream_transcript_from_chunks,
# )

# app = FastAPI(title="Humility Interview Backend", version="3.2")

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], allow_credentials=True,
#     allow_methods=["*"], allow_headers=["*"],
# )

# # Serve our JS/STT script
# app.mount(
#     "/static",
#     StaticFiles(directory=Path(__file__).parent / "static"),
#     name="static"
# )

# @app.get("/", tags=["Status"])
# def health():
#     return {"status": "online"}

# # ────────────────── TTS ─────────────────────────────────────────────
# @app.post(
#     "/generate_audio",
#     response_model=GenerateAudioResponse,
#     tags=["Audio Processing"]
# )
# async def generate_audio(req: GenerateAudioRequest):
#     if not req.text:
#         raise HTTPException(400, "Text is required")
#     audio = await get_speech_from_text(req.text)
#     if not audio:
#         raise HTTPException(500, "TTS failed")
#     return {"audio_b64": base64.b64encode(audio).decode()}

# # ────────────────── AI Analysis ─────────────────────────────────────
# @app.post(
#     "/analyze",
#     response_model=AnalysisResponse,
#     tags=["AI Analysis"]
# )
# async def analyze(req: AnalyzeRequest):
#     if not req.transcript:
#         raise HTTPException(400, "Transcript cannot be empty")
#     try:
#         res = await run_analysis_pipeline(req.transcript)
#         return {"scores": {
#             r.agent_name: {"score": r.score, "evidence": r.evidence}
#             for r in res
#         }}
#     except Exception as e:
#         print("Pipeline error:", e)
#         raise HTTPException(500, "Analysis failed")

# # ────────────────── File-based STT fallback ─────────────────────────
# def _maybe_to_wav(src: str) -> str:
#     """If src is MP3, convert to WAV @16 kHz mono via ffmpeg, else return src."""
#     if src.lower().endswith(".wav"):
#         return src
#     wav = f"{src}.wav"
#     subprocess.check_call([
#         "ffmpeg", "-y", "-i", src,
#         "-ar", "16000", "-ac", "1", wav
#     ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     return wav

# @app.post(
#     "/transcribe",
#     response_model=dict,
#     tags=["Audio Processing"]
# )
# async def transcribe(file: UploadFile = File(...)):
#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix)
#     tmp.write(await file.read()); tmp.flush(); tmp.close()
#     wav_path = _maybe_to_wav(tmp.name)
#     text = get_transcript_from_audio_sync(Path(wav_path).read_bytes())
#     os.unlink(tmp.name)
#     if wav_path != tmp.name:
#         os.unlink(wav_path)
#     if not text:
#         raise HTTPException(500, "Transcription failed")
#     return {"transcript": text}

# # ────────────────── Live STT streaming ──────────────────────────────
# @app.websocket("/stt_stream")
# async def stt_stream(ws: WebSocket):
#     """
#     Bidirectional streaming STT:
#     - Client sends raw 16 kHz mono PCM chunks (~200 ms per ArrayBuffer)
#     - Server sends JSON {id, partial, final}
#     """
#     await ws.accept()
#     aggregator = stream_transcript_from_chunks()
#     next(aggregator)  # prime the generator
#     try:
#         while True:
#             chunk = await ws.receive_bytes()
#             for update in aggregator.send(chunk):
#                 await ws.send_text(json.dumps(update))
#     except WebSocketDisconnect:
#         aggregator.close()
#     except Exception as e:
#         await ws.close(code=1011, reason=str(e))