# streamlit_realtime_transcribe.py

import io
import threading
import time

import numpy as np
import soundfile as sf
import streamlit as st
from faster_whisper import WhisperModel
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

st.set_page_config(page_title="Real‑Time ASR Demo", layout="wide")
st.title("🎙️ Local Real‑Time Speech‑to‑Text with ONNX Whisper")

# Load Whisper ONNX model once
@st.cache_resource
def load_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

model = load_model()

# Recorder to buffer PCM frames
class Recorder(AudioProcessorBase):
    def __init__(self):
        self.buffer = bytearray()
    def recv_audio(self, frame):
        # append raw int16 bytes
        self.buffer.extend(frame.to_ndarray().tobytes())
        return frame
    def get_wav_bytes(self):
        # convert raw bytes → float32 PCM → WAV
        arr = np.frombuffer(self.buffer, dtype=np.int16).astype(np.float32) / 32768.0
        buf = io.BytesIO()
        sf.write(buf, arr, 48000, format="WAV")
        return buf.getvalue()
    def clear(self):
        self.buffer.clear()

# UI placeholders
placeholder_controls = st.empty()
placeholder_audio = st.empty()
placeholder_transcript = st.empty()

recorder = None
ctx = None
running = False

def record_thread():
    global recorder, ctx, running
    while running:
        time.sleep(1)
        # you could snapshot intermediate transcripts here

# Controls
with placeholder_controls.container():
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Recording"):
            # clear old data
            placeholder_transcript.empty()
            placeholder_audio.empty()
            recorder = Recorder()
            ctx = webrtc_streamer(
                key="example",
                mode=WebRtcMode.SENDONLY,
                audio_processor_factory=lambda: recorder,
                media_stream_constraints={"audio": True, "video": False},
                async_processing=False
            )
            running = True
            threading.Thread(target=record_thread, daemon=True).start()
    with col2:
        if st.button("⏹️ Stop Recording"):
            running = False
            if recorder:
                wav_bytes = recorder.get_wav_bytes()
                # Playback
                placeholder_audio.audio(wav_bytes, format="audio/wav")
                # Transcribe
                segments, _ = model.transcribe(
                    io.BytesIO(wav_bytes), beam_size=1, language="en"
                )
                text = " ".join(seg.text for seg in segments)
                placeholder_transcript.markdown("**Transcript:**\n\n" + text)
                recorder.clear()

st.markdown(
    """
    **How it works**  
    - Click **Start Recording**, speak into your mic.  
    - Click **Stop Recording** to finish.  
    - The app plays back your audio and shows the transcript below.  
    - All processing is local with ONNX Whisper (`faster-whisper`).
    """
)
