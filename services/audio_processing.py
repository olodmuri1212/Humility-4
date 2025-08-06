# # services/audio_processing.py

# import whisper
# import tempfile
# import os
# from typing import Optional

# # Initialize Whisper model (will download on first use)
# print("Loading Whisper model for speech recognition...")
# WHISPER_MODEL = whisper.load_model("base")  # Use 'base' for good balance of speed/accuracy
# print("Whisper model loaded successfully.")

# def get_transcript_from_audio(audio_file_path: str) -> Optional[str]:
#     """
#     Uses local OpenAI Whisper to transcribe audio from a file.

#     Args:
#         audio_file_path: The local path to the audio file (e.g., .wav, .mp3).

#     Returns:
#         The transcribed text as a string, or None if an error occurs.
#     """
#     print(f"INFO: Transcribing audio file: {audio_file_path}")
#     try:
#         # Use Whisper to transcribe the audio
#         result = WHISPER_MODEL.transcribe(audio_file_path)
#         transcript = result["text"].strip()
        
#         if transcript:
#             print(f"INFO: Transcription successful. Transcript: '{transcript[:50]}...'")
#         else:
#             print("WARNING: Transcription resulted in an empty string.")
#         return transcript
        
#     except Exception as e:
#         print(f"ERROR: An error occurred during transcription: {e}")
#         return None


# async def get_speech_from_text(text: str) -> Optional[bytes]:
#     """
#     Uses espeak-ng directly to generate speech from text.
#     Bypasses pyttsx3 to avoid configuration issues.

#     Args:
#         text: The text to be converted to speech.

#     Returns:
#         The audio content as bytes, or None if an error occurs.
#     """
#     print(f"INFO: Generating speech for text: '{text}'")
#     try:
#         # Use espeak-ng to generate audio data directly to stdout
#         import subprocess
        
#         # Run espeak-ng to generate audio data directly to stdout
#         command = [
#             'espeak-ng',
#             '-s', '150',      # Speech rate (words per minute)
#             '-a', '100',      # Amplitude (volume)
#             '--stdout',       # Output audio to stdout
#             text
#         ]
        
#         result = subprocess.run(command, capture_output=True)
        
#         if result.returncode != 0:
#             print(f"ERROR: espeak-ng failed: {result.stderr.decode()}")
#             return None
            
#         # Get audio data directly from stdout
#         audio_bytes = result.stdout
        
#         if not audio_bytes:
#             print("ERROR: No audio data generated")
#             return None
            
#         print("INFO: Speech audio generated successfully.")
#         return audio_bytes

#     except Exception as e:
#         print(f"ERROR: An error occurred during speech generation: {e}")
#         return None











# """
# Pipecat‑powered STT & TTS helpers.
# Replace the espeak + blocking‑Whisper code in the original file.
# """
# from functools import lru_cache
# from typing import Optional, AsyncGenerator

# import asyncio
# try:
#     from pipecat.frames import TextFrame, TTSAudioRawFrame  # Pipecat ≥ 0.4
# except ImportError:  # Fallback for ≤ 0.3 where frames sub-package layout differs
#     from pipecat.frames.frames import TextFrame, TTSAudioRawFrame
# from pipecat.services.whisper.stt import WhisperSTTService
# from pipecat.services.elevenlabs.tts import ElevenLabsTTSService


# # ---------------------------------------------------------------------------
# #  STT  ── Whisper streamed through Pipecat
# # ---------------------------------------------------------------------------

# @lru_cache(maxsize=1)
# def _stt() -> WhisperSTTService:
#     # 'base' ≈ 500 MB VRAM.  Use 'small' / GPU for accuracy, or 'tiny' for speed.
#     return WhisperSTTService(model="base")


# async def get_transcript_from_audio(audio_bytes: bytes) -> Optional[str]:
#     """
#     Transcribe an in‑memory audio clip (wav/mp3/ogg) and return text.
#     """
#     try:
#         text = await _stt().transcribe_bytes(audio_bytes)
#         return text.strip() if text else None
#     except Exception as exc:
#         print("STT error:", exc)
#         return None


# # ---------------------------------------------------------------------------
# #  TTS  ── ElevenLabs streaming
# # ---------------------------------------------------------------------------

# @lru_cache(maxsize=1)
# def _tts() -> ElevenLabsTTSService:
#     # Set a default voice or tweak stability/style here if you like.
#     return ElevenLabsTTSService()  # uses ELEVENLABS_API_KEY env‑var


# async def get_speech_from_text(text: str) -> Optional[bytes]:
#     """
#     Stream ElevenLabs audio for `text` and return the raw PCM bytes.
#     """
#     try:
#         pcm_chunks: list[bytes] = []
#         async for frame in _tts().stream(text):
#             if isinstance(frame, TTSAudioRawFrame):
#                 pcm_chunks.append(frame.audio)
#         return b"".join(pcm_chunks) if pcm_chunks else None
#     except Exception as exc:
#         print("TTS error:", exc)
#         return None










# # services/audio_processing.py
# import os, asyncio
# from functools import lru_cache
# from typing import Optional
# from pipecat.services.whisper.stt import WhisperSTTService
# from pipecat.services.elevenlabs.tts import ElevenLabsTTSService


    
# """
# Pipecat‑powered STT & TTS helpers.
# Drop‑in replacement for your original Whisper+espeak file. :contentReference[oaicite:3]{index=3}
# """
# from functools import lru_cache
# from typing import Optional

# import asyncio
# from pipecat.services.whisper.stt import WhisperSTTService
# from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
# try:
#     from pipecat.frames import TextFrame, TTSAudioRawFrame  # Pipecat ≥ 0.4
# except ImportError:
#     from pipecat.frames.frames import TextFrame, TTSAudioRawFrame  # Pipecat ≤ 0.3

# # ——— STT ——————————————————————————————————————————————————————————

# @lru_cache(maxsize=1)
# def _stt() -> WhisperSTTService:
#     # 'base' ≈500 MB VRAM; switch to 'small'/'medium' on GPU
#     return WhisperSTTService(model="base")

# async def get_transcript_from_audio(audio_bytes: bytes) -> Optional[str]:
#     try:
#         text = await _stt().transcribe_bytes(audio_bytes)
#         return text.strip() if text else None
#     except Exception as e:
#         print("STT error:", e)
#         return None

# # ——— TTS ——————————————————————————————————————————————————————————

# # @lru_cache(maxsize=1)
# # def _tts() -> ElevenLabsTTSService:
# #     # Uses ELEVENLABS_API_KEY env var
# #     return ElevenLabsTTSService()

# # async def get_speech_from_text(text: str) -> Optional[bytes]:
# #     try:
# #         pcm: list[bytes] = []
# #         async for frame in _tts().stream(text):
# #             if isinstance(frame, TTSAudioRawFrame):
# #                 pcm.append(frame.audio)
# #         return b"".join(pcm) if pcm else None
# #     except Exception as e:
# #         print("TTS error:", e)
# #         return None



# @lru_cache(maxsize=1)
# def _tts() -> ElevenLabsTTSService:
#     key = os.getenv("ELEVENLABS_API_KEY")
#     if not key:
#         print("⚠️  ELEVENLABS_API_KEY is not set!")
#     # Pass it explicitly so we know it’s used:
#     return ElevenLabsTTSService(api_key=key)

# async def get_speech_from_text(text: str) -> Optional[bytes]:
#     try:
#         pcm = []
#         async for frame in _tts().stream(text):
#             if isinstance(frame, TTSAudioRawFrame):
#                 pcm.append(frame.audio)
#         if not pcm:
#             print("⚠️  ElevenLabs returned *no* audio frames for:", text[:30])
#             return None
#         return b"".join(pcm)
#     except Exception as e:
#         print("🔥 TTS exception:", repr(e))
#         return None



















# services/audio_processing.py
# """
# Local real‑time ASR & TTS powered by ONNX Runtime.

# Dependencies
# ============
# pip install onnxruntime soundfile numpy librosa TTS==0.21.4
# """
# import io
# from functools import lru_cache
# from typing import Optional, Tuple

# import numpy as np
# import soundfile as sf
# #import whisper_onnx
# from TTS.api import TTS

# ########################################################################
# #                          SPEECH → TEXT  (Whisper‑ONNX)
# ########################################################################


# from faster_whisper import WhisperModel

# @lru_cache(maxsize=1)
# def _stt_model():
#     return WhisperModel("base", device="cpu", compute_type="int8")

# async def get_transcript_from_audio(audio_bytes):
#     pcm, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
#     # resample to 16 kHz mono as before…
#     segments, _ = _stt_model().transcribe(pcm, beam_size=1)
#     return " ".join(seg.text for seg in segments).strip()



# # @lru_cache(maxsize=1)
# # def _stt_model():
# #     # Loads the base.en model (~145 MB cached under ~/.cache/whisper_onnx/)
# #     return whisper_onnx.load_model(
# #         "base.en", device="cpu", intra_op_num_threads=2
# #     )

# # def _decode_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
# #     pcm, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
# #     if sr != 16000:
# #         import librosa
# #         pcm = librosa.resample(pcm.T, orig_sr=sr, target_sr=16000).T
# #         sr = 16000
# #     # mono mix if needed
# #     if pcm.ndim > 1:
# #         pcm = pcm.mean(axis=1)
# #     return pcm, sr

# # async def get_transcript_from_audio(audio_bytes: bytes) -> Optional[str]:
# #     try:
# #         pcm, _ = _decode_audio(audio_bytes)
# #         result = _stt_model().transcribe(pcm, language="en")
# #         text = result["text"].strip()
# #         return text or None
# #     except Exception as e:
# #         print("STT error:", e)
# #         return None

# ########################################################################
# #                          TEXT → SPEECH (Coqui‑TTS ONNX)
# ########################################################################
# @lru_cache(maxsize=1)
# def _tts_model():
#     # Loads & caches the VITS model (~50 MB under ~/.local/share/tts/)
#     return TTS(
#         model_name="tts_models/en/vctk/vits",
#         progress_bar=False,
#         gpu=False,
#         backend="onnx"
#     )

# async def get_speech_from_text(text: str) -> Optional[bytes]:
#     try:
#         wav = _tts_model().tts(text)
#         buf = io.BytesIO()
#         sf.write(buf, wav, _tts_model().sample_rate, format="WAV")
#         return buf.getvalue()
#     except Exception as e:
#         print("TTS error:", e)
#         return None




























# # services/audio_processing.py
# """
# Local real‑time ASR & TTS powered by faster‑whisper (ONNXRuntime under the hood)
# and Coqui‑TTS (v0.22.x) without unsupported kwargs.

# Dependencies
# ============
# pip install onnxruntime soundfile numpy librosa faster-whisper TTS==0.22.0
# """
# import io
# from functools import lru_cache
# from typing import Optional, Tuple

# import numpy as np
# import soundfile as sf

# # faster-whisper for STT
# from faster_whisper import WhisperModel
# # Coqui‑TTS for TTS
# from TTS.api import TTS

# # ─── SPEECH → TEXT (faster‑whisper) ────────────────────────────────────────

# @lru_cache(maxsize=1)
# def _asr_model() -> WhisperModel:
#     # 'base' for CPU. Switch to 'small' or add device="cuda" if you have a GPU.
#     return WhisperModel("base", device="cpu", compute_type="int8")

# def _decode_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
#     """Load any wav/ogg/mp3 into float32 PCM @16 kHz mono."""
#     pcm, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
#     if sr != 16000:
#         import librosa
#         pcm = librosa.resample(pcm.T, sr, 16000).T
#         sr = 16000
#     # stereo → mono
#     if pcm.ndim > 1:
#         pcm = pcm.mean(axis=1)
#     return pcm, sr

# async def get_transcript_from_audio(audio_bytes: bytes) -> Optional[str]:
#     try:
#         pcm, _ = _decode_audio(audio_bytes)
#         segments, _ = _asr_model().transcribe(pcm, beam_size=1)
#         text = " ".join(seg.text.strip() for seg in segments)
#         return text or None
#     except Exception as e:
#         print("STT error:", e)
#         return None

# # ─── TEXT → SPEECH (Coqui‑TTS v0.22.x) ─────────────────────────────────────

# # @lru_cache(maxsize=1)
# # def _tts_model() -> TTS:
# #     # Note: no 'backend' argument any more in v0.22.x
# #     return TTS(
# #         model_name="tts_models/en/vctk/vits",
# #         progress_bar=False,
# #         gpu=False
# #     )
# @lru_cache(maxsize=1)
# def _tts_model() -> TTS:
#     return TTS(
#         model_name="tts_models/en/vctk/vits",
#         progress_bar=False,
#         gpu=False,
#         phonemizer="g2p_en"        # <-- switch away from espeak
#     )

# async def get_speech_from_text(text: str) -> Optional[bytes]:
#     try:
#         wav = _tts_model().tts(text)
#         buf = io.BytesIO()
#         sf.write(buf, wav, _tts_model().sample_rate, format="WAV")
#         return buf.getvalue()
#     except Exception as e:
#         print("TTS error:", e)
#         return None














# # services/audio_processing.py
# """
# Local real‑time ASR & TTS powered by:
#  - faster‑whisper (ONNX Runtime) for STT
#  - Coqui‑TTS (v0.22.x) for TTS, with automatic speaker selection

# Dependencies
# ============
# # ONNX STT
# pip install onnxruntime soundfile numpy librosa faster-whisper[onnxruntime]

# # Coqui TTS
# pip install "git+https://github.com/coqui-ai/TTS.git@v0.22.0"

# # (system) phonemizer backend
# sudo apt update && sudo apt install -y espeak-ng
# """
# import io
# from functools import lru_cache
# from typing import Optional, Tuple

# import numpy as np
# import soundfile as sf

# # faster‑whisper for STT
# from faster_whisper import WhisperModel
# # Coqui‑TTS for TTS
# from TTS.api import TTS

# # ─── SPEECH → TEXT (faster‑whisper) ────────────────────────────────────────

# @lru_cache(maxsize=1)
# def _asr_model() -> WhisperModel:
#     return WhisperModel("base", device="cpu", compute_type="int8")

# def _decode_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
#     pcm, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
#     if sr != 16000:
#         import librosa
#         pcm = librosa.resample(pcm.T, sr, 16000).T
#         sr = 16000
#     if pcm.ndim > 1:
#         pcm = pcm.mean(axis=1)
#     return pcm, sr

# async def get_transcript_from_audio(audio_bytes: bytes) -> Optional[str]:
#     try:
#         pcm, _ = _decode_audio(audio_bytes)
#         segments, _ = _asr_model().transcribe(pcm, beam_size=1)
#         text = " ".join(seg.text.strip() for seg in segments)
#         return text or None
#     except Exception as e:
#         print("STT error:", e)
#         return None

# # ─── TEXT → SPEECH (Coqui‑TTS v0.22.x) ─────────────────────────────────────

# @lru_cache(maxsize=1)
# def _tts_model() -> TTS:
#     # Loads the VCTK VITS multi‑speaker model (~50 MB on first run).
#     return TTS(
#         model_name="tts_models/en/vctk/vits",
#         progress_bar=False,
#         gpu=False
#     )

# async def get_speech_from_text(text: str) -> Optional[bytes]:
#     try:
#         model = _tts_model()
#         # Pick the first speaker if it's a multi‑speaker model
#         speaker = None
#         if hasattr(model, "speakers") and model.speakers:
#             speaker = model.speakers[0]

#         # Synthesize
#         wav = model.tts(text, speaker=speaker) if speaker else model.tts(text)

#         # Determine sample rate
#         if hasattr(model, "output_sample_rate"):
#             sr = model.output_sample_rate
#         elif hasattr(model, "synthesizer") and hasattr(model.synthesizer, "output_sample_rate"):
#             sr = model.synthesizer.output_sample_rate
#         else:
#             # Fallback to the documented default for VCTK‑VITS
#             sr = 22050

#         # Write WAV in‑memory
#         buf = io.BytesIO()
#         sf.write(buf, wav, sr, format="WAV")
#         return buf.getvalue()

#     except Exception as e:
#         print("TTS error:", e)
#         return None




















# # services/audio_processing.py
# """
# Local real‑time ASR & TTS powered by:
#  - faster‑whisper (ONNX Runtime) for STT
#  - Coqui‑TTS (v0.22.x) for TTS

# Dependencies
# ============
# # ONNX STT
# pip install onnxruntime soundfile numpy faster-whisper[onnxruntime]

# # Coqui TTS
# pip install "git+https://github.com/coqui-ai/TTS.git@v0.22.0"

# # (system) phonemizer
# sudo apt update && sudo apt install -y espeak-ng
# """
# import io
# import tempfile
# from functools import lru_cache
# from typing import Optional

# import soundfile as sf
# from faster_whisper import WhisperModel
# from TTS.api import TTS

# # ─── SPEECH → TEXT (faster‑whisper) ────────────────────────────────────────

# @lru_cache(maxsize=1)
# def _asr_model() -> WhisperModel:
#     return WhisperModel("base", device="cpu", compute_type="int8")

# async def get_transcript_from_audio(audio_bytes: bytes) -> Optional[str]:
#     try:
#         # Write bytes to a temp WAV so faster-whisper handles loading & resampling
#         with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
#             tmp.write(audio_bytes)
#             tmp.flush()
#             segments, _ = _asr_model().transcribe(tmp.name, beam_size=1)
#         text = " ".join(seg.text.strip() for seg in segments)
#         return text or None
#     except Exception as e:
#         print("STT error:", e)
#         return None

# # ─── TEXT → SPEECH (Coqui‑TTS v0.22.x) ─────────────────────────────────────

# @lru_cache(maxsize=1)
# def _tts_model() -> TTS:
#     return TTS(
#         model_name="tts_models/en/vctk/vits",
#         progress_bar=False,
#         gpu=False
#     )

# async def get_speech_from_text(text: str) -> Optional[bytes]:
#     try:
#         model = _tts_model()
#         # Pick first speaker if multi‑speaker
#         speaker = getattr(model, "speakers", [None])[0]
#         wav = model.tts(text, speaker=speaker) if speaker else model.tts(text)

#         # Coqui's VITS default sample rate is 22050
#         sr = getattr(model, "output_sample_rate", 22050)

#         buf = io.BytesIO()
#         sf.write(buf, wav, sr, format="WAV")
#         return buf.getvalue()
#     except Exception as e:
#         print("TTS error:", e)
#         return None


# # services/audio_processing.py  (bottom of file)
# def get_transcript_from_audio_sync(audio_bytes: bytes) -> str | None:
#     import asyncio
#     return asyncio.run(get_transcript_from_audio(audio_bytes))




# # ─── Streaming STT helper ───────────────────────────────────────────
# import numpy as np
# from itertools import islice

# FRAME_SIZE = 16000 // 5          # 0.2 s of audio @16 kHz
# BYTES_PER_FRAME = FRAME_SIZE * 2 # 16‑bit (2 bytes)

# def stream_transcript_from_chunks():
#     """
#     Coroutine‑style generator.
#       • .send(pcm_bytes)  -> yields a list[dict] of partial/final updates
#       • .close()          -> flush & return remaining finals
#     Keeps an internal rolling buffer so Whisper can see context.
#     """
#     model = _asr_model()
#     buffer = bytearray()
#     seg_id = 0
#     while True:
#         chunk = (yield [])              # accept bytes
#         if chunk is None:               # flush signal
#             break
#         buffer.extend(chunk)

#         # Only run inference when we have ≥ 1 s new audio
#         while len(buffer) >= 5 * BYTES_PER_FRAME:
#             # Take the first N bytes => 1 s window, keep overlap for context
#             wav_bytes = bytes(buffer[:10 * BYTES_PER_FRAME])  # 2 s sliding window
#             del buffer[:5 * BYTES_PER_FRAME]                  # 1 s stride

#             with tempfile.NamedTemporaryFile(suffix=".wav") as f:
#                 # raw PCM → WAVE container
#                 wf = wave.open(f, "wb")
#                 wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
#                 wf.writeframes(wav_bytes); wf.close()
#                 segments, _ = model.transcribe(f.name,
#                                                beam_size=1,
#                                                vad_filter=True,
#                                                patience=1)
#             outs = []
#             for s in segments:
#                 outs.append({"id": seg_id,
#                              "partial": s.text.strip(),
#                              "final": True})
#                 seg_id += 1
#             yield outs





















#claud ai 
"""
Enhanced Audio Processing Service
Real-time ASR & TTS with improved error handling and performance

Dependencies:
============
pip install faster-whisper soundfile numpy librosa
pip install "git+https://github.com/coqui-ai/TTS.git@v0.22.0"

System Requirements:
===================
sudo apt update && sudo apt install -y espeak-ng ffmpeg
"""

import io
import tempfile
import asyncio
import logging
from functools import lru_cache
from typing import Optional, Union, Tuple
from pathlib import Path

import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import TTS with error handling
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TTS not available: {e}")
    TTS_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# SPEECH-TO-TEXT (Enhanced faster-whisper implementation)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    """Get cached Whisper model instance"""
    try:
        logger.info("Loading Whisper model (base)...")
        model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8",
            download_root=None,
            local_files_only=False
        )
        logger.info("Whisper model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise

def preprocess_audio(audio_data: bytes) -> Tuple[np.ndarray, int]:
    """
    Preprocess audio data for better STT results
    Returns: (audio_array, sample_rate)
    """
    try:
        # Try to load audio with soundfile
        with io.BytesIO(audio_data) as audio_buffer:
            audio, sr = sf.read(audio_buffer, dtype=np.float32)
        
        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Ensure minimum length (avoid very short clips)
        min_length = int(0.1 * sr)  # 100ms minimum
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)), mode='constant')
        
        return audio, sr
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        raise

async def get_transcript_from_audio(audio_bytes: bytes) -> Optional[str]:
    """
    Enhanced async speech-to-text using faster-whisper
    """
    if not audio_bytes or len(audio_bytes) < 100:
        logger.warning("Audio data too short or empty")
        return None
    
    try:
        logger.info(f"Starting STT for {len(audio_bytes)} bytes of audio")
        
        # Preprocess audio
        audio, sr = preprocess_audio(audio_bytes)
        
        # Create temporary file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sr, format='WAV')
            tmp_path = tmp_file.name
        
        try:
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(
                None, 
                _transcribe_file, 
                tmp_path
            )
            
            if transcript and transcript.strip():
                logger.info(f"STT success: '{transcript[:50]}...'")
                return transcript.strip()
            else:
                logger.warning("STT returned empty transcript")
                return None
                
        finally:
            # Cleanup temp file
            try:
                Path(tmp_path).unlink()
            except:
                pass
                
    except Exception as e:
        logger.error(f"STT processing failed: {e}")
        return None

def _transcribe_file(file_path: str) -> str:
    """
    Internal function to transcribe file (runs in thread pool)
    """
    try:
        model = get_whisper_model()
        
        # Transcribe with optimized settings
        segments, info = model.transcribe(
            file_path,
            beam_size=1,           # Faster decoding
            temperature=0.0,       # Deterministic
            condition_on_previous_text=False,  # Don't condition on previous
            initial_prompt=None,   # No prompt
            word_timestamps=False, # Don't need word timestamps
            vad_filter=True,      # Use voice activity detection
            vad_parameters=dict(
                min_silence_duration_ms=500,
                threshold=0.5
            )
        )
        
        # Combine segments
        text_segments = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                text_segments.append(text)
        
        result = " ".join(text_segments)
        logger.debug(f"Transcription completed: {len(text_segments)} segments")
        
        return result
        
    except Exception as e:
        logger.error(f"File transcription failed: {e}")
        return ""

def get_transcript_from_audio_sync(audio_bytes: bytes) -> Optional[str]:
    """
    Synchronous wrapper for STT (for compatibility)
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(get_transcript_from_audio(audio_bytes))
    except Exception as e:
        logger.error(f"Sync STT failed: {e}")
        return None
    finally:
        try:
            loop.close()
        except:
            pass

# ──────────────────────────────────────────────────────────────────────────────
# TEXT-TO-SPEECH (Enhanced Coqui-TTS implementation)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_tts_model():
    """Get cached TTS model instance"""
    if not TTS_AVAILABLE:
        raise RuntimeError("TTS not available - please install Coqui-TTS")
    
    try:
        logger.info("Loading TTS model...")
        model = TTS(
            model_name="tts_models/en/vctk/vits",
            progress_bar=False,
            gpu=False
        )
        logger.info("TTS model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load TTS model: {e}")
        raise

async def get_speech_from_text(text: str) -> Optional[bytes]:
    """
    Enhanced async text-to-speech using Coqui-TTS
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for TTS")
        return None
    
    if not TTS_AVAILABLE:
        logger.error("TTS not available")
        return None
    
    try:
        text = text.strip()
        logger.info(f"Starting TTS for text: '{text[:50]}...'")
        
        # Run TTS in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None,
            _synthesize_speech,
            text
        )
        
        if audio_data:
            logger.info(f"TTS success: Generated {len(audio_data)} bytes")
            return audio_data
        else:
            logger.warning("TTS returned no audio data")
            return None
            
    except Exception as e:
        logger.error(f"TTS processing failed: {e}")
        return None

def _synthesize_speech(text: str) -> Optional[bytes]:
    """
    Internal function to synthesize speech (runs in thread pool)
    """
    try:
        model = get_tts_model()
        
        # Get available speakers
        speakers = getattr(model, 'speakers', None)
        speaker = speakers[0] if speakers else None
        
        # Generate speech
        if speaker:
            wav = model.tts(text, speaker=speaker)
        else:
            wav = model.tts(text)
        
        # Get sample rate
        sample_rate = getattr(model, 'output_sample_rate', 22050)
        
        # Convert to WAV bytes
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, wav, sample_rate, format='WAV')
        wav_bytes = wav_buffer.getvalue()
        
        logger.debug(f"Speech synthesis completed: {len(wav_bytes)} bytes")
        return wav_bytes
        
    except Exception as e:
        logger.error(f"Speech synthesis failed: {e}")
        return None

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def validate_audio_format(audio_bytes: bytes) -> bool:
    """
    Validate if audio data is in a supported format
    """
    try:
        with io.BytesIO(audio_bytes) as buffer:
            sf.read(buffer, frames=1)
        return True
    except:
        return False

def get_audio_info(audio_bytes: bytes) -> dict:
    """
    Get information about audio data
    """
    try:
        with io.BytesIO(audio_bytes) as buffer:
            info = sf.info(buffer)
        
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype
        }
    except Exception as e:
        return {"error": str(e)}

async def test_audio_services() -> dict:
    """
    Test both STT and TTS services
    """
    results = {
        "stt": {"status": "unknown", "error": None},
        "tts": {"status": "unknown", "error": None}
    }
    
    # Test TTS
    try:
        test_text = "This is a test of the text to speech system."
        audio_data = await get_speech_from_text(test_text)
        if audio_data and len(audio_data) > 1000:
            results["tts"]["status"] = "working"
        else:
            results["tts"]["status"] = "failed"
            results["tts"]["error"] = "No audio generated"
    except Exception as e:
        results["tts"]["status"] = "error"
        results["tts"]["error"] = str(e)
    
    # Test STT (if TTS worked)
    if results["tts"]["status"] == "working":
        try:
            transcript = await get_transcript_from_audio(audio_data)
            if transcript and len(transcript.strip()) > 0:
                results["stt"]["status"] = "working"
                results["stt"]["transcript"] = transcript
            else:
                results["stt"]["status"] = "failed"
                results["stt"]["error"] = "No transcript generated"
        except Exception as e:
            results["stt"]["status"] = "error"
            results["stt"]["error"] = str(e)
    
    return results

def create_silent_audio(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """
    Create silent audio for testing
    """
    try:
        num_samples = int(duration_seconds * sample_rate)
        silence = np.zeros(num_samples, dtype=np.float32)
        
        buffer = io.BytesIO()
        sf.write(buffer, silence, sample_rate, format='WAV')
        return buffer.getvalue()
    except Exception as e:
        logger.error(f"Failed to create silent audio: {e}")
        return b""

# ──────────────────────────────────────────────────────────────────────────────
# INITIALIZATION AND HEALTH CHECKS
# ──────────────────────────────────────────────────────────────────────────────

async def initialize_audio_services():
    """
    Initialize and warm up audio services
    """
    logger.info("Initializing audio services...")
    
    # Warm up models
    try:
        # Load Whisper model
        get_whisper_model()
        logger.info("✅ STT service initialized")
    except Exception as e:
        logger.error(f"❌ STT initialization failed: {e}")
    
    try:
        # Load TTS model if available
        if TTS_AVAILABLE:
            get_tts_model()
            logger.info("✅ TTS service initialized")
        else:
            logger.warning("⚠️ TTS service not available")
    except Exception as e:
        logger.error(f"❌ TTS initialization failed: {e}")
    
    # Run service tests
    test_results = await test_audio_services()
    logger.info(f"Service test results: {test_results}")
    
    return test_results






















"""
Local real‑time ASR & TTS powered by:
 - faster‑whisper (ONNX Runtime) for STT
 - Coqui‑TTS (v0.22.x) for TTS
"""
import io
import tempfile
from functools import lru_cache
from typing import Optional

import soundfile as sf
from faster_whisper import WhisperModel
from TTS.api import TTS

# ─── SPEECH → TEXT (faster‑whisper) ─────────────────────────────────
@lru_cache(maxsize=1)
def _asr_model() -> WhisperModel:
    return WhisperModel("base", device="cpu", compute_type="int8")

async def get_transcript_from_audio(audio_bytes: bytes) -> Optional[str]:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            segments, _ = _asr_model().transcribe(tmp.name, beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments)
        return text or None
    except Exception as e:
        print("STT error:", e)
        return None

# ─── TEXT → SPEECH (Coqui‑TTS) ────────────────────────────────────────
@lru_cache(maxsize=1)
def _tts_model() -> TTS:
    return TTS(
        model_name="tts_models/en/vctk/vits",
        progress_bar=False,
        gpu=False
    )

async def get_speech_from_text(text: str) -> Optional[bytes]:
    try:
        model = _tts_model()
        speaker = getattr(model, "speakers", [None])[0]
        wav = model.tts(text, speaker=speaker) if speaker else model.tts(text)
        sr = getattr(model, "output_sample_rate", 22050)
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV")
        return buf.getvalue()
    except Exception as e:
        print("TTS error:", e)
        return None

# ─── Sync helper ────────────────────────────────────────────────────
def get_transcript_from_audio_sync(audio_bytes: bytes) -> Optional[str]:
    import asyncio
    return asyncio.run(get_transcript_from_audio(audio_bytes))

# ─── Streaming STT helper ───────────────────────────────────────────
import numpy as np
import wave

FRAME_SIZE = 16000 // 5          # 0.2 s @ 16kHz
BYTES_PER_FRAME = FRAME_SIZE * 2 # 16‑bit PCM

def stream_transcript_from_chunks():
    """
    Coroutine: .send(pcm_bytes) → yields list of {id, partial, final}
    """
    model = _asr_model()
    buffer = bytearray()
    seg_id = 0
    pcm = (yield)  # prime
    while True:
        chunk = pcm or b""
        buffer.extend(chunk)
        # Process every 1 s (5 * 0.2 s)
        while len(buffer) >= 5 * BYTES_PER_FRAME:
            window = bytes(buffer[:10 * BYTES_PER_FRAME])  # 2 s window
            del buffer[:5 * BYTES_PER_FRAME]
            with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                wf = wave.open(f, "wb")
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
                wf.writeframes(window); wf.close()
                segments, _ = model.transcribe(
                    f.name, beam_size=1, vad_filter=True, patience=1
                )
            updates = []
            for s in segments:
                updates.append({"id": seg_id, "partial": s.text.strip(), "final": True})
                seg_id += 1
            pcm = (yield updates)