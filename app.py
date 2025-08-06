# app.py

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import av
import httpx
import asyncio
import base64
import io
import tempfile
import os

from core.state import InterviewState, Turn, AgentScore
from core.scoring import calculate_normalized_score # Re-enable the correct import
from services.report_generator import create_pdf_report

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000" # URL of your FastAPI backend

# --- Page Setup ---
st.set_page_config(
    page_title="Humility Interview Bot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Humility Interview Bot")
st.markdown("This AI-powered interviewer assesses humility through a structured conversation. Your responses will be analyzed in real-time. Press 'Start Interview' to begin.")

# --- Audio Recorder Processor ---
# This class uses the streamlit-webrtc library to capture audio from the user's microphone.
class AudioRecorder(AudioProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self.audio_frames = []
        self._lock = asyncio.Lock()

    async def add_frame(self, frame: av.AudioFrame):
        async with self._lock:
            self.audio_frames.append(frame.to_ndarray())

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # This method is called for each audio frame. We process it asynchronously.
        asyncio.run(self.add_frame(frame))
        return frame

    async def get_audio_bytes(self) -> bytes:
        """Concatenates all captured audio frames and returns them as WAV bytes."""
        async with self._lock:
            if not self.audio_frames:
                return b""
            
            # Combine all frames and clear the buffer for the next recording
            audio_data = np.concatenate(self.audio_frames, axis=1)
            self.audio_frames.clear()
            
            # Convert to 16-bit PCM format, which is standard for WAV
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # Use PyAV to create a WAV file in memory
            buffer = io.BytesIO()
            with av.open(buffer, mode='w', format='wav') as container:
                stream = container.add_stream('pcm_s16le', rate=48000, layout='stereo')
                frame = av.AudioFrame.from_ndarray(audio_data, format='s16', layout='stereo')
                frame.rate = 48000
                for packet in container.encode(frame):
                    container.mux(packet)
            return buffer.getvalue()

# --- Session State Management ---
if "interview_state" not in st.session_state:
    st.session_state.interview_state = InterviewState()
if "recorder_active" not in st.session_state:
    st.session_state.recorder_active = False

# --- UI Layout ---
sidebar_col, main_col = st.columns([1, 2.5])

with sidebar_col:
    st.header("Live Score")
    score_placeholder = st.empty()
    score_placeholder.progress(0, text="Score: 0/100")
    
    st.markdown("---")
    st.header("Controls")
    control_placeholder = st.empty()
    
    st.markdown("---")
    st.info("Your conversation is private. Audio is processed in memory and not stored after the session ends.")

with main_col:
    st.header("Conversation")
    chat_placeholder = st.container(height=600)

# --- Backend API Client ---
async def api_call(method: str, endpoint: str, json_data: dict = None, timeout: float = 60.0):
    """A helper function to make asynchronous API calls to the FastAPI backend."""
    async with httpx.AsyncClient() as client:
        try:
            url = f"{BACKEND_URL}{endpoint}"
            if method.upper() == "POST":
                response = await client.post(url, json=json_data, timeout=timeout)
            else:
                response = await client.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except httpx.ReadTimeout:
            st.error(f"Request to '{endpoint}' timed out. The backend might be busy processing.")
        except httpx.ConnectError:
            st.error("Connection Error: Could not connect to the backend. Please ensure it's running.")
        except Exception as e:
            st.error(f"An API error occurred: {e}")
        return None

# --- Core Interview Logic ---
async def run_interview_turn():
    """Manages a single turn of the interview: asking a question and processing the answer."""
    state: InterviewState = st.session_state.interview_state
    
    # 1. Get the next question from the state
    if not state.mandatory_questions:
        state.interview_status = "finished"
        st.rerun()
        return
        
    question_text = state.mandatory_questions[0] # Peek at the next question

    # 2. Display question and generate/play audio
    with chat_placeholder.chat_message("assistant", avatar="🤖"):
        st.write(question_text)
        with st.spinner("Assistant is speaking..."):
            response = await api_call("POST", "/generate_audio", {"text": question_text})
            if response and "audio_b64" in response:
                try:
                    audio_bytes = base64.b64decode(response["audio_b64"])
                    
                    # Save to static directory for Streamlit to serve
                    import time
                    timestamp = int(time.time() * 1000)  # milliseconds
                    audio_filename = f"question_{timestamp}.wav"
                    audio_path = os.path.join("static", audio_filename)
                    
                    # Ensure static directory exists
                    os.makedirs("static", exist_ok=True)
                    
                    # Write audio file
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)
                    
                    # Try both st.audio and HTML5 audio for better compatibility
                    st.audio(audio_path, format="audio/wav")
                    
                    # Also try HTML5 audio with autoplay
                    st.markdown(f"""
                    <audio controls autoplay>
                        <source src="{audio_path}" type="audio/wav">
                        Your browser does not support the audio element.
                    </audio>
                    <p><strong>📢 If you don't hear audio, click the play button above</strong></p>
                    """, unsafe_allow_html=True)
                    
                    print(f"DEBUG: Audio generated successfully, {len(audio_bytes)} bytes")
                    print(f"DEBUG: Audio saved to {audio_path}")
                    
                except Exception as e:
                    print(f"ERROR: Failed to play audio: {e}")
                    st.warning("Audio generation succeeded but playback failed.")

    # 3. Activate the audio recorder
    st.session_state.recorder_active = True
    st.rerun()
ry, I couldn't understand that. Could you please try again?")
            state.mandatory_questions.insert(0, question_text) # Put question back
            return

        transcript = transcribe_response["transcript"]
        
        # 5. Analyze transcript
        analysis_response = await api_call("POST", "/analyze", {"transcript": transcript, "question": question_text})
        if not analysis_response:
            st.error("Sorry, there was an error analyzing your response.")
            return

        # 6. Update state
        current_turn = Turn(question=question_text, transcript=transcript)
        for agent, result in analysis_response.get("scores", {}).items():
            score = AgentScore(agent_name=agent, score=result['score'], evidence=result['evidence'])
            current_turn.analysis_results.append(score)
            state.cumulative_scores[agent] = state.cumulative_scores.get(agent, 0) + result['score']
        
        state.conversation_history.append(current_turn)
        state.normalized_humility_score = calculate_normalized_score(state.cumulative_scores)

    st.rerun()

# --- UI Rendering and Control Flow ---

# Display conversation history
for turn in st.session_state.interview_state.conversation_history:
    with chat_placeholder.chat_message("assistant", avatar="🤖"):
        st.write(turn.question)
    with chat_placeholder.chat_message("user"):
        st.write(turn.transcript)

# Main control flow based on interview status
status = st.session_state.interview_state.interview_status

if status == "not_started":
    if control_placeholder.button("▶️ Start Interview"):
        st.session_state.interview_state.interview_status = "in_progress"
        asyncio.run(run_interview_turn())

elif status == "in_progress" and st.session_state.recorder_active:
    # Show the recorder with clear instructions
    with main_col:
        st.info("🎤 **Recording your response...** Please speak clearly into your microphone.")
        st.success("🔴 Recording is ACTIVE - Click 'Stop Recording' when finished")
    
    ctx = webrtc_streamer(
        key="recorder",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioRecorder,
        medi
async def process_recorded_audio(audio_bytes: bytes):
    """Handles transcription and analysis of the user's recorded audio."""
    state: InterviewState = st.session_state.interview_state
    question_text = state.mandatory_questions.pop(0) # Officially ask the question

    with st.spinner("Analyzing your response... This may take a moment."):
        # 4. Transcribe audio
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        transcribe_response = await api_call("POST", "/transcribe", {"audio_b64": audio_b64})
        
        if not transcribe_response or not transcribe_response.get("transcript"):
            st.error("Sora_stream_constraints={"video": False, "audio": True},
    )
    
    if control_placeholder.button("⏹️ Stop Recording", type="primary"):
        st.session_state.recorder_active = False
        with st.spinner("Processing your response..."):
            if ctx.audio_processor:
                audio_bytes = asyncio.run(ctx.audio_processor.get_audio_bytes())
                if audio_bytes:
                    st.success(f"✅ Audio captured successfully! ({len(audio_bytes)} bytes)")
                    asyncio.run(process_recorded_audio(audio_bytes))
                else:
                    st.warning("⚠️ No audio was detected. Please try speaking again.")
                    st.session_state.recorder_active = True  # Allow retry
                    st.rerun()
            else:
                st.error("❌ Audio recorder was not ready. Please refresh and try again.")
                st.rerun()

elif status == "in_progress" and not st.session_state.recorder_active:
    # If not recording, it means a turn just finished, so start the next one.
    asyncio.run(run_interview_turn())

elif status == "finished":
    with main_col:
        st.success("🎉 Interview Complete!")
        st.balloons()
    
    # Generate and offer the report for download
    pdf_bytes = create_pdf_report(st.session_state.interview_state)
    control_placeholder.download_button(
        label="⬇️ Download Full Report (PDF)",
        data=pdf_bytes,
        file_name=f"humility_report_{st.session_state.interview_state.interview_id[:8]}.pdf",
        mime="application/pdf",
    )

# Always update the score display
score = st.session_state.interview_state.normalized_humility_score
score_placeholder.progress(score / 100, text=f"Score: {score}/100")
