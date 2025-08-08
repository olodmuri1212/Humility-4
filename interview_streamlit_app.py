import os
import json
import base64
import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import numpy as np
import threading
import queue

# Import from existing modules
from services.report_generator import create_pdf_report
from interview_analyzer import InterviewAnalyzer, InterviewTurn

# Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

# Initialize Whisper model for transcription
try:
    from faster_whisper import WhisperModel
    WHISPER_MODEL = WhisperModel("base")  # Use "small" or "medium" for better accuracy
except ImportError:
    WHISPER_MODEL = None
    st.warning("Whisper model not available. Install with: pip install faster-whisper")

# Interview questions bank
QUESTION_BANK = [
    "Can you tell me about a time when you received constructive criticism? How did you handle it?",
    "Describe a situation where you had to work with a difficult team member. How did you handle it?",
    "Tell me about a time when you made a mistake at work. How did you address it?",
    "How do you handle situations where you need to learn something new?",
    "Can you share an example of when you had to adapt to a significant change at work?"
]

# Initialize session state
def init_session_state():
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'recorded_answers' not in st.session_state:
        st.session_state.recorded_answers = {}
    if 'transcripts' not in st.session_state:
        st.session_state.transcripts = {}
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'interview_analyzer' not in st.session_state:
        st.session_state.interview_analyzer = InterviewAnalyzer()
    if 'show_report' not in st.session_state:
        st.session_state.show_report = False
    if 'transcription' not in st.session_state:
        st.session_state.transcription = ""
    if 'transcription_queue' not in st.session_state:
        st.session_state.transcription_queue = queue.Queue()

# Audio processor class
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []
        self.recording = False
        self.transcription = ""
    
    def recv(self, frame):
        if self.recording:
            # Convert audio frame to numpy array
            audio_data = frame.to_ndarray()
            self.audio_frames.append(audio_data)
            
            # Add to transcription queue if model is available
            if WHISPER_MODEL and len(self.audio_frames) % 10 == 0:  # Process every 10 chunks
                audio_np = np.concatenate(self.audio_frames[-10:])  # Get last 10 chunks
                st.session_state.transcription_queue.put(audio_np)
                
        return frame
    
    def start_recording(self):
        self.audio_frames = []
        self.recording = True
        st.session_state.transcription = ""
    
    def stop_recording(self):
        self.recording = False
        if self.audio_frames:
            return np.concatenate(self.audio_frames)
        return None

# Background thread for transcription
def process_transcription():
    while True:
        if not st.session_state.transcription_queue.empty():
            audio_data = st.session_state.transcription_queue.get()
            try:
                # Transcribe audio using Whisper
                segments, _ = WHISPER_MODEL.transcribe(
                    audio_data, 
                    language="en",
                    beam_size=5
                )
                for segment in segments:
                    st.session_state.transcription += segment.text + " "
            except Exception as e:
                st.error(f"Transcription error: {str(e)}")

# Start transcription thread if not already running
if 'transcription_thread' not in st.session_state and WHISPER_MODEL:
    st.session_state.transcription_thread = threading.Thread(target=process_transcription, daemon=True)
    st.session_state.transcription_thread.start()

# UI Components
def show_question(question_idx: int):
    st.subheader(f"Question {question_idx + 1} of {len(QUESTION_BANK)}")
    st.markdown(f"**{QUESTION_BANK[question_idx]}**")
    
    # Initialize audio processor in session state
    if 'audio_processor' not in st.session_state:
        st.session_state.audio_processor = AudioRecorder()
    
    # Audio recording section
    st.markdown("### 🎤 Record Your Response")
    
    # WebRTC streamer for audio capture
    webrtc_ctx = webrtc_streamer(
        key=f"audio-recorder-{question_idx}",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioRecorder,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎤 Start Recording", key=f"start_{question_idx}"):
            if webrtc_ctx.audio_processor:
                webrtc_ctx.audio_processor.start_recording()
                st.session_state.recording = True
    
    with col2:
        if st.button("⏹️ Stop Recording", key=f"stop_{question_idx}"):
            if webrtc_ctx.audio_processor and st.session_state.get('recording', False):
                audio_data = webrtc_ctx.audio_processor.stop_recording()
                if audio_data is not None:
                    st.session_state.recorded_answers[question_idx] = audio_data
                    st.session_state.transcripts[question_idx] = st.session_state.transcription
                    st.session_state.transcription = ""  # Reset for next recording
                st.session_state.recording = False
    
    # Show current recording status
    if st.session_state.get('recording', False):
        st.warning("Recording in progress...")
        
        # Show real-time transcription
        if WHISPER_MODEL:
            st.markdown("**Live Transcription:**")
            transcription_placeholder = st.empty()
            transcription_placeholder.text_area("Transcription", 
                                             value=st.session_state.transcription, 
                                             height=100,
                                             disabled=True)
        else:
            st.info("Install faster-whisper for real-time transcription")
    
    # Show saved transcript if available
    if question_idx in st.session_state.transcripts:
        st.markdown("**Your response:**")
        st.text_area("Response", 
                    value=st.session_state.transcripts[question_idx], 
                    height=150,
                    disabled=True,
                    key=f"transcript_{question_idx}")

def show_navigation():
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.session_state.current_question > 0:
            if st.button("⬅️ Previous"):
                st.session_state.current_question -= 1
                st.rerun()
    
    with col2:
        if st.session_state.current_question < len(QUESTION_BANK) - 1:
            if st.button("Next ➡️"):
                st.session_state.current_question += 1
                st.rerun()
    
    with col3:
        if st.button("📊 Generate Report", 
                    disabled=len(st.session_state.transcripts) < len(QUESTION_BANK)):
            st.session_state.show_report = True
            st.rerun()

def show_report():
    st.title("📊 Interview Report")
    
    # Generate analysis for each question
    for i, question in enumerate(QUESTION_BANK):
        if i in st.session_state.transcripts:
            st.subheader(f"Question {i+1}")
            st.markdown(f"**{question}**")
            st.markdown(f"*Your response:* {st.session_state.transcripts[i]}")
            
            # Here you would show analysis for each response
            st.markdown("""
            **Analysis:** 
            - This is where the detailed analysis would appear
            - Including scores for different dimensions
            - And specific feedback on the response
            """)
            st.markdown("---")
    
    # Add a button to download the full report
    if st.button("📥 Download Full Report"):
        # In a real app, this would generate a PDF using create_pdf_report
        st.success("Report generated! (This is a placeholder - implement PDF generation)")

# Main app
def main():
    st.set_page_config(
        page_title="AI Interview Assistant",
        page_icon="🎙️",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Add custom CSS
    st.markdown("""
    <style>
        .main-header { color: #1E88E5; }
        .stButton>button { width: 100%; }
    </style>
    """, unsafe_allow_html=True)
    
    # Main title
    st.markdown("<h1 class='main-header'>🎙️ AI Interview Assistant</h1>", unsafe_allow_html=True)
    st.markdown("Record your interview responses and get instant feedback on your answers.")
    
    # Show interview progress
    progress = (st.session_state.current_question / len(QUESTION_BANK)) * 100
    st.progress(int(progress))
    st.caption(f"Question {st.session_state.current_question + 1} of {len(QUESTION_BANK)}")
    
    # Show current question or report
    if not st.session_state.show_report:
        show_question(st.session_state.current_question)
        show_navigation()
    else:
        show_report()
        
        if st.button("🔄 Start New Interview"):
            # Reset the interview
            for key in ['current_question', 'recorded_answers', 'transcripts', 
                       'analysis_results', 'recording', 'show_report', 'transcription']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
