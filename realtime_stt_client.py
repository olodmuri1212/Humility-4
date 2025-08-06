# """
# Streamlit page: live microphone → 1‑second WAV chunks → POST /transcribe →
# append transcript.

# Run with:
#     streamlit run realtime_stt_client.py
# Requires:
#     streamlit-webrtc, requests
# """

# import asyncio, base64, io, requests
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# BACKEND = "http://127.0.0.1:8000"       # <- your existing FastAPI server

# st.set_page_config(page_title="Live STT (faster‑whisper via backend)")
# st.title("🎤 Real‑time Speech‑to‑Text (backend faster‑whisper)")

# TRANSCRIPT_KEY = "live_transcript"
# st.session_state.setdefault(TRANSCRIPT_KEY, "")

# # --------------------- WebRTC capture ---------------------

# CHUNK_MS = 1000           # send 1‑second chunks

# class Segmenter(AudioProcessorBase):
#     """Collect PCM frames and yield 1‑second WAV chunks."""
#     def __init__(self):
#         self.buf = bytearray()
#         self.samplerate = 48000   # WebRTC default
#         self.last_flush = asyncio.get_event_loop().time()

#     def recv_audio(self, frame):
#         self.buf.extend(frame.to_ndarray().tobytes())
#         now = asyncio.get_event_loop().time()
#         if (now - self.last_flush) * 1000 >= CHUNK_MS:
#             audio = bytes(self.buf)
#             self.buf.clear()
#             self.last_flush = now
#             asyncio.run_coroutine_threadsafe(send_chunk(audio), asyncio.get_event_loop())
#         return frame

# async def send_chunk(raw_pcm: bytes):
#     """Encode WAV + POST to /transcribe, append result to transcript."""
#     if not raw_pcm:
#         return
#     # Wrap raw 48 kHz mono int16 PCM into a WAV header in‑memory
#     import soundfile as sf, numpy as np, io
#     pcm = np.frombuffer(raw_pcm, dtype="int16")
#     wav_buf = io.BytesIO()
#     sf.write(wav_buf, pcm, 48000, format="WAV")
#     wav_b64 = base64.b64encode(wav_buf.getvalue()).decode()

#     try:
#         r = requests.post(f"{BACKEND}/transcribe", json={"audio_b64": wav_b64}, timeout=30)
#         r.raise_for_status()
#         text = r.json().get("transcript", "")
#         if text:
#             st.session_state[TRANSCRIPT_KEY] += " " + text
#             placeholder.markdown(st.session_state[TRANSCRIPT_KEY])
#     except Exception as e:
#         st.error(f"STT chunk failed: {e}")

# placeholder = st.empty()

# webrtc_streamer(
#     key="live-stt",
#     mode=WebRtcMode.SENDONLY,
#     audio_processor_factory=Segmenter,
#     media_stream_constraints={"audio": True, "video": False},
#     async_processing=True
# )









import asyncio
import base64
import io
import requests
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

BACKEND = "http://127.0.0.1:8000"
st.title("Live STT via faster‑whisper backend")

# live transcript
st.session_state.setdefault("transcript", "")
placeholder = st.empty()

class Segmenter(AudioProcessorBase):
    def __init__(self):
        self.buf = bytearray()
        self.last = asyncio.get_event_loop().time()
    def recv_audio(self, frame):
        self.buf.extend(frame.to_ndarray().tobytes())
        now = asyncio.get_event_loop().time()
        if (now - self.last) > 1.0:  # every 1 sec
            data = bytes(self.buf)
            self.buf.clear()
            self.last = now
            asyncio.run_coroutine_threadsafe(self.send_chunk(data), asyncio.get_event_loop())
        return frame

    async def send_chunk(self, pcm_bytes):
        # wrap PCM into WAV
        import soundfile as sf, numpy as np
        pcm = np.frombuffer(pcm_bytes, dtype="int16")
        buf = io.BytesIO()
        sf.write(buf, pcm, 48000, format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode()
        r = requests.post(f"{BACKEND}/transcribe", json={"audio_b64": b64})
        if r.ok:
            txt = r.json().get("transcript", "")
            st.session_state.transcript += " " + txt
            placeholder.markdown(st.session_state.transcript)

webrtc_streamer(
    key="client-stt",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=Segmenter,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)
