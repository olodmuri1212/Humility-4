import os
import base64
import json
import requests
import streamlit as st
import soundfile as sf
import numpy as np
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, List
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase

# Configuration
BACKEND = "http://localhost:8000"  # Update this if your backend is running elsewhere

# Set page config
st.set_page_config(
    page_title="AI-Powered Interview Analysis",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state variables
if 'question' not in st.session_state:
    st.session_state.question = "Tell me about a time you were wrong."
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'analysis' not in st.session_state:
    st.session_state.analysis = None
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = "auto"  # Default to auto mode
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'interim_transcript' not in st.session_state:
    st.session_state.interim_transcript = ""
if 'final_transcript' not in st.session_state:
    st.session_state.final_transcript = ""
if 'speech_error' not in st.session_state:
    st.session_state.speech_error = ""

# App header
st.markdown("<h1 class='main-header'>🎙️ AI Interview Assistant</h1>", unsafe_allow_html=True)
st.markdown("Record your interview responses and get instant feedback on your communication skills.")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.question = st.text_area(
        "Interview Question",
        value=st.session_state.question,
        height=100
    )
    
    mode = st.radio(
        "Analysis Mode",
        ["Auto (Record Audio)", "Manual (Enter Text)", "Speech-to-Text (Beta)"],
        index=0 if st.session_state.analysis_mode == "auto" else (1 if st.session_state.analysis_mode == "manual" else 2)
    )
    if mode == "Auto (Record Audio)":
        st.session_state.analysis_mode = "auto"
    elif mode == "Manual (Enter Text)":
        st.session_state.analysis_mode = "manual"
    else:
        st.session_state.analysis_mode = "speech"
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This tool helps you practice for interviews by analyzing your responses for key qualities like:
    - Humility
    - Learning Mindset
    - Feedback Seeking
    - Mistake Handling
    
    Choose between automatic audio recording or manual text input for analysis.
    """)

def analyze_transcript(transcript: str, question: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Send transcript to backend for analysis."""
    try:
        response = requests.post(
            f"{BACKEND}/analyze",
            json={
                "transcript": transcript,
                "question": question
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"Analysis failed with status {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"Error during analysis: {str(e)}"

def transcribe_audio(audio_b64: str) -> Tuple[Optional[str], Optional[str]]:
    """Send audio to backend for transcription."""
    try:
        response = requests.post(
            f"{BACKEND}/transcribe",
            json={"audio_b64": audio_b64},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("transcript", ""), None
        else:
            return None, f"Transcription failed with status {response.status_code}: {response.text}"
    except Exception as e:
        return None, f"Error during transcription: {str(e)}"

# Audio Recorder Component
class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []
        self.sample_rate = 16000
        
    def recv_audio(self, frame):
        self.audio_frames.append(frame.to_ndarray())
        return frame
    
    def get_audio_data(self) -> Optional[Tuple[np.ndarray, int]]:
        if not self.audio_frames:
            return None
        
        try:
            # Combine all audio frames
            audio_array = np.concatenate(self.audio_frames, axis=0)
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                audio_array = np.mean(audio_array, axis=1)
                
            # Normalize audio
            if np.abs(audio_array).max() > 0:
                audio_array = audio_array / np.abs(audio_array).max()
                
            return audio_array, self.sample_rate
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            return None

def handle_speech_messages():
    if 'transcript' in st.query_params:
        try:
            msg = json.loads(st.query_params['transcript'])
            if msg.get('isStreamlitMessage'):
                if msg.get('type') == 'transcript':
                    st.session_state.interim_transcript = msg.get('interim', '')
                    st.session_state.final_transcript = msg.get('final', '')
                    st.session_state.transcript = st.session_state.final_transcript
                    st.rerun()
                elif msg.get('type') == 'recordingState':
                    st.session_state.is_recording = msg.get('isRecording', False)
                    st.rerun()
                elif msg.get('type') == 'speechError':
                    error_msg = msg.get('error', '')
                    st.session_state.speech_error = error_msg if error_msg else None
                    if error_msg:
                        st.toast(f"Speech recognition error: {error_msg}", icon="⚠️")
                    st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"Error processing message: {e}")

# Main content columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 🎤 " + ("Record Your Response" if st.session_state.analysis_mode == "auto" else "Enter Your Response"))
    
    if st.session_state.analysis_mode == "auto":
        # Audio recording section
        st.markdown("Click the 'Start Recording' button below and speak your answer to the question.")
        
        # Initialize WebRTC streamer
        webrtc_ctx = webrtc_streamer(
            key="audio-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_processor_factory=AudioRecorder,
            media_stream_constraints={"audio": True, "video": False},
            async_processing=True,
        )
        
        # Process recorded audio when recording stops
        if webrtc_ctx.audio_processor and st.button("Stop Recording & Analyze"):
            with st.spinner("Processing your response..."):
                # Get audio data
                audio_processor = webrtc_ctx.audio_processor
                audio_data = audio_processor.get_audio_data()
                
                if audio_data:
                    # Save audio to bytes
                    audio_array, sample_rate = audio_data
                    buffer = BytesIO()
                    sf.write(buffer, audio_array, sample_rate, format='WAV')
                    audio_bytes = buffer.getvalue()
                    
                    # Convert to base64 for API
                    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                    
                    # Transcribe audio
                    transcript, error = transcribe_audio(audio_b64)
                    
                    if transcript is not None:
                        st.session_state.transcript = transcript
                        st.session_state.audio_data = audio_bytes
                        
                        # Analyze transcript
                        analysis, error = analyze_transcript(transcript, st.session_state.question)
                        
                        if analysis is not None:
                            st.session_state.analysis = analysis
                            st.success("Analysis complete!")
                        else:
                            st.error(f"Analysis failed: {error}")
                    else:
                        st.error(f"Transcription failed: {error}")
                else:
                    st.warning("No audio data recorded. Please try again.")
    elif st.session_state.analysis_mode == "manual":
        # Manual text input section
        st.session_state.transcript = st.text_area(
            "Enter your response:",
            value=st.session_state.transcript,
            height=200,
            key="manual_transcript"
        )
        
        if st.button("Analyze Response") and st.session_state.transcript.strip():
            with st.spinner("Analyzing your response..."):
                # Analyze transcript
                analysis, error = analyze_transcript(
                    st.session_state.transcript, 
                    st.session_state.question
                )
                
                if analysis is not None:
                    st.session_state.analysis = analysis
                    st.success("Analysis complete!")
                else:
                    st.error(f"Analysis failed: {error}")
        elif st.button("Analyze Response") and not st.session_state.transcript.strip():
            st.warning("Please enter a response to analyze.")
    else:
        # Speech-to-Text mode using Web Speech API
        st.markdown("### 🎤 Speak Your Response")
        
        # Show the current question before recording
        st.markdown("### ❓ Interview Question")
        st.markdown(f"<div style='background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem;'>{st.session_state.question}</div>", 
                   unsafe_allow_html=True)
        
        # Status indicator
        status_placeholder = st.empty()
        if st.session_state.is_recording:
            status_placeholder.markdown(
                "<div style='background-color: #ffebee; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #f44336;'>"
                "<div style='display: flex; align-items: center;'>"
                "<span style='display: inline-block; width: 12px; height: 12px; background-color: #f44336; border-radius: 50%; margin-right: 8px; animation: pulse 1.5s infinite;'></span>"
                "<span style='color: #c62828; font-weight: 500;'>Recording in progress...</span>"
                "</div>"
                "<div style='margin-top: 8px; color: #5d4037; font-size: 0.9em;'>"
                "Speak clearly into your microphone. Click 'Stop Recording' when finished."
                "</div>"
                "<style>@keyframes pulse { 0% { opacity: 0.6; } 50% { opacity: 1; } 100% { opacity: 0.6; } }</style>",
                unsafe_allow_html=True
            )
        else:
            status_placeholder.markdown(
                "<div style='background-color: #e8f5e9; padding: 0.8rem; border-radius: 0.5rem; margin-bottom: 1rem; border-left: 4px solid #4caf50;'>"
                "<div style='display: flex; align-items: center;'>"
                "<span style='color: #2e7d32; font-weight: 500;'>Ready to record</span>"
                "</div>"
                "<div style='margin-top: 8px; color: #1b5e20; font-size: 0.9em;'>"
                "Click 'Start Recording' and speak your answer. Click 'Stop Recording' when done."
                "</div>"
                "</div>",
                unsafe_allow_html=True
            )
        
        # Display the recorded text
        if st.session_state.final_transcript:
            st.markdown("### 🎙️ Your Recorded Response")
            st.markdown(f"<div style='background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;'>{st.session_state.final_transcript}</div>", 
                      unsafe_allow_html=True)
        
        # Add Web Speech API implementation
        components.html(f"""
        <div>
            <button id="startButton" class="button" {'disabled' if st.session_state.is_recording else ''}>
                🎤 Start Recording
            </button>
            <button id="stopButton" class="button" {'disabled' if not st.session_state.is_recording else ''}>
                ⏹️ Stop Recording
            </button>
            <div id="interim" style="color: #666; font-style: italic; min-height: 1.5rem; margin: 0.5rem 0;">
                {st.session_state.interim_transcript or ''}
            </div>
            <div id="final" style="display: none;"></div>
        </div>
        <style>
            .button {{
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 0.5rem 1rem;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 1rem;
                margin: 0.25rem 0.5rem 0.25rem 0;
                cursor: pointer;
                border-radius: 4px;
                transition: background-color 0.3s;
            }}
            .button:disabled {{
                background-color: #cccccc;
                cursor: not-allowed;
            }}
            #startButton:not(:disabled) {{
                background-color: #4CAF50;
            }}
            #stopButton:not(:disabled) {{
                background-color: #f44336;
            }}
            #startButton:not(:disabled):hover {{
                background-color: #45a049;
            }}
            #stopButton:not(:disabled):hover {{
                background-color: #da190b;
            }}
        </style>
        <script>
            const startButton = document.getElementById('startButton');
            const stopButton = document.getElementById('stopButton');
            const interimEl = document.getElementById('interim');
            const finalEl = document.getElementById('final');
            let recognition;
            let isFinalizing = false;
            
            // Function to send message to Streamlit
            function sendToStreamlit(message) {{
                window.parent.postMessage({{
                    isStreamlitMessage: true,
                    ...message
                }}, '*');
                
                // Also update the URL with the transcript
                const url = new URL(window.location);
                url.searchParams.set('transcript', JSON.stringify({{
                    isStreamlitMessage: true,
                    ...message
                }}));
                window.history.pushState({{}}, '', url);
            }}
            
            // Check for browser support
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {{
                interimEl.innerHTML = '❌ Web Speech API is not supported in this browser. Please use Chrome, Edge, or Safari.';
                startButton.disabled = true;
                stopButton.disabled = true;
            }} else {{
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SpeechRecognition();
                recognition.continuous = true;
                recognition.interimResults = true;
                
                // Handle results
                recognition.onresult = (event) => {{
                    let interim = '';
                    let final = '';
                    
                    // Process all results
                    for (let i = event.resultIndex; i < event.results.length; i++) {{
                        const transcript = event.results[i][0].transcript;
                        if (event.results[i].isFinal) {{
                            final += transcript + ' ';
                        }} else {{
                            interim += transcript;
                        }}
                    }}
                    
                    // Update the UI
                    interimEl.textContent = interim || 'Listening...';
                    
                    // Send to Streamlit
                    if (final) {{
                        finalEl.textContent += final;
                        sendToStreamlit({{
                            type: 'transcript',
                            interim: interim,
                            final: finalEl.textContent
                        }});
                    }} else if (interim) {{
                        sendToStreamlit({{
                            type: 'transcript',
                            interim: interim,
                            final: finalEl.textContent
                        }});
                    }}
                }};
                
                // Handle errors
                recognition.onerror = (event) => {{
                    console.error('Speech recognition error', event.error);
                    let errorMessage = 'An error occurred with speech recognition.';
                    
                    // Map error codes to user-friendly messages
                    switch(event.error) {{
                        case 'not-allowed':
                            errorMessage = 'Microphone access was denied. Please allow microphone access to use this feature.';
                            break;
                        case 'audio-capture':
                            errorMessage = 'No microphone was found. Please ensure a microphone is connected.';
                            break;
                        case 'language-not-supported':
                            errorMessage = 'The selected language is not supported.';
                            break;
                        case 'service-not-allowed':
                            errorMessage = 'Speech recognition service is not allowed.';
                            break;
                        case 'network':
                            errorMessage = 'Network error occurred. Please check your internet connection.';
                            break;
                        case 'no-speech':
                            errorMessage = 'No speech was detected. Please try again.';
                            break;
                        default:
                            errorMessage = `Error: ${{event.error}}`;
                    }}
                    
                    // Send error to Streamlit
                    sendToStreamlimessage({{type: 'speechError', error: errorMessage}});
                    
                    // Update button states
                    startButton.disabled = false;
                    stopButton.disabled = true;
                    isFinalizing = false;
                }};
                
                // Handle end of recognition
                recognition.onend = () => {{
                    if (!isFinalizing) {{
                        // If we're not finalizing, restart recognition
                        try {{
                            recognition.start();
                        }} catch (e) {{
                            console.error('Error restarting recognition:', e);
                            startButton.disabled = false;
                            stopButton.disabled = true;
                            
                            sendToStreamlit({{
                                type: 'recordingState',
                                isRecording: false
                            }});
                        }}
                    }} else {{
                        // We're done finalizing
                        startButton.disabled = false;
                        stopButton.disabled = true;
                        isFinalizing = false;
                        
                        // Notify Streamlit that recording has stopped
                        sendToStreamlit({{
                            type: 'recordingState',
                            isRecording: false
                        }});
                    }}
                }};
                
                // Button event listeners
                startButton.onclick = () => {{
                    try {{
                        // Clear any previous error
                        sendToStreamlimessage({{type: 'speechError', error: ''}});
                        
                        // Clear previous transcripts
                        finalEl.textContent = '';
                        interimEl.textContent = 'Listening...';
                        
                        // Start recognition
                        recognition.start();
                        startButton.disabled = true;
                        stopButton.disabled = false;
                        
                        // Notify Streamlit that recording has started
                        sendToStreamlimessage({{type: 'recordingState', isRecording: true}});
                    }} catch (error) {{
                        console.error('Error starting speech recognition:', error);
                        sendToStreamlimessage({{
                            type: 'speechError',
                            error: 'Failed to start speech recognition. Please try again.'
                        }});
                    }}
                }};
                
                stopButton.onclick = () => {{
                    try {{
                        isFinalizing = true;
                        recognition.stop();
                        interimEl.textContent = 'Processing...';
                        stopButton.disabled = true;
                    }} catch (error) {{
                        console.error('Error stopping speech recognition:', error);
                        startButton.disabled = false;
                        stopButton.disabled = true;
                        isFinalizing = false;
                        
                        sendToStreamlimessage({{
                            type: 'speechError',
                            error: 'Error stopping speech recognition.'
                        }});
                    }}
                }};
            }}
        </script>
        """, height=200)
        
        # Add analyze button for speech mode
        if st.session_state.final_transcript:
            if st.button("📊 Analyze Response"):
                with st.spinner("Analyzing your response..."):
                    # Analyze the final transcript
                    analysis, error = analyze_transcript(
                        st.session_state.final_transcript,
                        st.session_state.question
                    )
                    
                    if analysis is not None:
                        st.session_state.analysis = analysis
                        st.session_state.transcript = st.session_state.final_transcript
                        st.success("Analysis complete!")
                        st.rerun()
                    else:
                        st.error(f"Analysis failed: {error}")
        elif st.button("📊 Analyze Response"):
            st.warning("Please record your response first.")

# Call the message handler
handle_speech_messages()

# Clear error message when switching modes
if st.session_state.analysis_mode != "speech" and st.session_state.speech_error:
    st.session_state.speech_error = ""

# Display results
with col2:
    if st.session_state.transcript:
        st.markdown("### 📝 Transcript")
        st.markdown(f"<div class='success-box'>{st.session_state.transcript}</div>", unsafe_allow_html=True)
        
        # Play recorded audio if available
        if st.session_state.analysis_mode == "auto" and st.session_state.audio_data:
            st.audio(st.session_state.audio_data, format="audio/wav")
        
        if st.session_state.analysis:
            st.markdown("### 📊 Analysis Results")
            
            # Handle different response formats from the backend
            if isinstance(st.session_state.analysis, dict):
                # New format with scores dictionary
                scores = st.session_state.analysis.get("scores", {})
                
                # Display overall score if available
                overall_score = st.session_state.analysis.get("overall_score")
                if overall_score is not None:
                    if isinstance(overall_score, (int, float)):
                        st.metric("Overall Score", f"{overall_score * 100:.1f}%")
                
                # Display category scores
                if scores:
                    for category, score_info in scores.items():
                        if isinstance(score_info, dict):
                            # Handle dictionary format with score and evidence
                            score = score_info.get("score", 0)
                            evidence = score_info.get("evidence", "")
                            
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.metric(category.title(), f"{score * 100:.1f}%")
                            with col2:
                                st.progress(float(score))
                            
                            if evidence:
                                with st.expander(f"View evidence for {category}"):
                                    st.write(evidence)
                        else:
                            # Handle simple score format
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.metric(category.title(), f"{score_info * 100:.1f}%")
                            with col2:
                                st.progress(float(score_info))
                
                # Display detailed feedback if available
                if "feedback" in st.session_state.analysis:
                    st.markdown("### 📝 Detailed Feedback")
                    feedback = st.session_state.analysis["feedback"]
                    if isinstance(feedback, list):
                        for item in feedback:
                            st.markdown(f"- {item}")
                    else:
                        st.markdown(f"<div class='metric-box'>{feedback}</div>", unsafe_allow_html=True)
                
                # Display suggestions for improvement if available
                if "suggestions" in st.session_state.analysis and st.session_state.analysis["suggestions"]:
                    st.markdown("### 💡 Suggestions for Improvement")
                    suggestions = st.session_state.analysis["suggestions"]
                    if isinstance(suggestions, list):
                        for suggestion in suggestions:
                            st.markdown(f"- {suggestion}")
                    else:
                        st.markdown(suggestions)
            
            # Fallback for unexpected formats
            elif st.session_state.analysis:
                st.markdown("### Analysis Results")
                st.json(st.session_state.analysis)
    
    elif st.session_state.analysis_mode == "manual":
        st.info("Enter your response in the left panel and click 'Analyze Response' to see results here.")

# Add JavaScript to handle Web Speech API messages
st.components.v1.html("""
<script>
// Listen for messages from the Web Speech API component
window.addEventListener('message', function(event) {
    // Only process messages from our own iframe
    if (event.origin !== window.location.origin) return;
    
    const msg = event.data;
    if (msg && msg.isStreamlitMessage) {
        // Forward the message to Streamlit
        window.parent.postMessage({
            type: 'streamlit:setQueryParams',
            data: {
                msg: JSON.stringify(msg)
            }
        }, '*');
    }
});
</script>
""", height=0)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #2c3e50; margin-bottom: 1rem;}
    .sub-header {font-size: 1.5rem; color: #3498db; margin-top: 1.5rem;}
    .success-box {
        background-color: #e8f5e9; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        margin: 1rem 0;
        white-space: pre-wrap;
    }
    .metric-box {
        background-color: #f5f5f5; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        margin: 0.5rem 0;
        white-space: pre-wrap;
    }
    </style>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
*This tool is for practice purposes only. Results may vary based on audio quality and content.*

**Note:** The analysis is based on AI models and may not be 100% accurate. 
Use it as a guide to improve your interview skills.
""")