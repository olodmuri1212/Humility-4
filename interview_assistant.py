# import gradio as gr
# import asyncio
# import json
# import os
# import sys
# import tempfile
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple, Any
# import numpy as np
# import torch
# import pyttsx3
# from datetime import datetime

# # Try to import Whisper model
# try:
#     from faster_whisper import WhisperModel
#     WHISPER_AVAILABLE = True
# except ImportError:
#     print("Warning: faster_whisper not found. Using fallback transcription.")
#     WHISPER_AVAILABLE = False

# # Initialize Whisper model
# if WHISPER_AVAILABLE:
#     try:
#         print("Loading Whisper model (base) on cpu...")
#         whisper_model = WhisperModel(
#             "base",
#             device="cpu",
#             compute_type="int8"
#         )
#         print("Whisper model loaded successfully!")
#     except Exception as e:
#         print(f"Error loading Whisper model: {e}")
#         WHISPER_AVAILABLE = False

# sys.path.append(str(Path(__file__).parent))

# try:
#     from backend.agent_manager import AnalysisResult, run_analysis_pipeline
# except ImportError as e:
#     print(f"Warning: Could not import backend agents: {e}")
#     print("Using fallback analysis functions")
    
#     from typing import NamedTuple
    
#     class AnalysisResult(NamedTuple):
#         """Fallback AnalysisResult class"""
#         humility_score: float
#         humility_evidence: str
#         learning_score: float
#         learning_evidence: str
#         feedback_score: float
#         feedback_evidence: str
#         mistakes_score: float
#         mistakes_evidence: str
    
#     async def run_analysis_pipeline(transcript: str) -> AnalysisResult:
#         """Fallback analysis pipeline"""
#         return AnalysisResult(
#             humility_score=0.0,
#             humility_evidence="Analysis not available",
#             learning_score=0.0,
#             learning_evidence="Analysis not available",
#             feedback_score=0.0,
#             feedback_evidence="Analysis not available",
#             mistakes_score=0.0,
#             mistakes_evidence="Analysis not available"
#         )

# # Interview questions
# QUESTIONS = [
#     "Can you tell me about a time when you received constructive criticism? How did you handle it?",
#     "Describe a situation where you had to work with a difficult team member. How did you handle it?",
#     "Tell me about a time when you made a mistake at work. How did you address it?",
#     "How do you handle situations where you need to learn something new?",
#     "Can you share an example of when you had to adapt to a significant change at work?"
# ]

# class InterviewState:
#     def __init__(self):
#         self.current_question = 0
#         self.answers = {}  # {question_index: {'text': str, 'analysis': dict}}
#         self.complete = False
#         self.candidate_name = ""
#         self.start_time = datetime.now()
#         self.scores = {
#             'humility': 0,
#             'learning': 0,
#             'feedback': 0,
#             'mistakes': 0
#         }
#         self.evidence = {}
#         self.recording_in_progress = False

# # Global state
# state = InterviewState()

# async def analyze_response(question_index: int, transcript: str) -> dict:
#     """Analyze the response using all available agents asynchronously."""
#     if not transcript or not transcript.strip():
#         return {}
    
#     try:
#         # Run the analysis pipeline asynchronously
#         analysis_result = await run_analysis_pipeline(transcript)
        
#         # Convert the analysis result to a dictionary format
#         analysis = {
#             'humility': {
#                 'score': analysis_result.humility_score,
#                 'evidence': analysis_result.humility_evidence
#             },
#             'learning': {
#                 'score': analysis_result.learning_score,
#                 'evidence': analysis_result.learning_evidence
#             },
#             'feedback': {
#                 'score': analysis_result.feedback_score,
#                 'evidence': analysis_result.feedback_evidence
#             },
#             'mistakes': {
#                 'score': analysis_result.mistakes_score,
#                 'evidence': analysis_result.mistakes_evidence
#             }
#         }
        
#         # Update overall scores
#         for category in ['humility', 'learning', 'feedback', 'mistakes']:
#             state.scores[category] = (state.scores[category] * question_index + analysis[category]['score']) / (question_index + 1)
        
#         return analysis
        
#     except Exception as e:
#         print(f"Error in analyze_response: {e}")
#         return {}

# def generate_interview_report() -> str:
#     """Generate a comprehensive HTML report of the interview."""
#     if not state.answers:
#         return "<p>No interview data available.</p>"
    
#     # Calculate overall score (average of all category scores)
#     overall_score = sum(state.scores.values()) / len(state.scores) if state.scores else 0
    
#     # Prepare data for the report
#     report_data = {
#         'candidate_name': state.candidate_name,
#         'date': state.start_time.strftime('%Y-%m-%d %H:%M:%S'),
#         'duration': str(datetime.now() - state.start_time).split('.')[0],  # Remove microseconds
#         'overall_score': overall_score,
#         'category_scores': state.scores,
#         'questions': [],
#         'recommendations': []
#     }
    
#     # Add question-answer pairs with analysis
#     for i, question in enumerate(QUESTIONS[:len(state.answers)]):
#         answer_data = state.answers.get(i, {
#             'text': 'No answer provided.',
#             'analysis': {}
#         })
        
#         report_data['questions'].append({
#             'question': question,
#             'answer': answer_data.get('text', 'No answer provided.'),
#             'analysis': answer_data.get('analysis', {})
#         })
    
#     # Generate recommendations based on scores
#     if overall_score < 5:
#         report_data['recommendations'].append("Consider working on self-awareness and openness to feedback.")
#     if state.scores['learning'] < 5:
#         report_data['recommendations'].append("Focus on developing a growth mindset and continuous learning.")
#     if state.scores['feedback'] < 5:
#         report_data['recommendations'].append("Practice seeking and accepting constructive feedback.")
    
#     # Generate HTML report
#     return generate_html_report(report_data)

# def transcribe_audio(audio_path):
#     """Transcribe audio data using the Whisper model or fallback to simple processing."""
#     if not audio_path or not os.path.exists(audio_path):
#         print(f"Audio file not found: {audio_path}")
#         return "[No audio file found]"
    
#     try:
#         # Read the audio file using soundfile
#         try:
#             import soundfile as sf
#             audio_data, sample_rate = sf.read(audio_path)
            
#             # Convert to mono if needed
#             if len(audio_data.shape) > 1:
#                 audio_data = np.mean(audio_data, axis=1)
                
#             # Convert to float32 if needed
#             if audio_data.dtype != np.float32:
#                 audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
                
#         except Exception as e:
#             print(f"Error reading audio file {audio_path}: {e}")
#             return f"[Error reading audio: {str(e)}]"
            
#         if WHISPER_AVAILABLE:
#             try:
#                 # Transcribe using Whisper
#                 segments, _ = whisper_model.transcribe(
#                     audio_data,
#                     language="en",
#                     beam_size=5,
#                     vad_filter=True
#                 )
#                 transcription = " ".join(segment.text.strip() for segment in segments).strip()
#                 if transcription:
#                     return transcription
#                 return "[No speech detected]"
                    
#             except Exception as e:
#                 print(f"Whisper transcription failed: {e}")
        
#         return "[Audio transcribed but no text detected]"
        
#     except Exception as e:
#         print(f"Error during audio processing: {e}")
#         return f"[Error processing audio: {str(e)}]"

# async def process_response(audio_path):
#     """Process the recorded response and update the interview state."""
#     if state.complete:
#         return "", "Interview completed. Please download your report.", "", ""
    
#     # Transcribe the audio
#     transcript = transcribe_audio(audio_path)
#     if not transcript:
#         return "", "Could not transcribe audio. Please try again.", "", ""
    
#     # Store the answer
#     state.answers[state.current_question] = {
#         'text': transcript,
#         'analysis': {}
#     }
    
#     # Analyze the response
#     analysis = await analyze_response(state.current_question, transcript)
#     state.answers[state.current_question]['analysis'] = analysis
    
#     # Prepare feedback
#     feedback = []
#     for agent, result in analysis.items():
#         feedback.append(f"{agent}: {result.get('evidence', 'No specific feedback')}")
    
#     feedback_text = "\n\n".join(feedback) if feedback else "No specific feedback available."
    
#     # Move to next question or complete
#     state.current_question += 1
    
#     if state.current_question >= len(QUESTIONS):
#         state.complete = True
#         next_question = "Interview complete! Click 'View Report' to see your results."
#         next_btn = "View Report"
#     else:
#         next_question = f"Question {state.current_question + 1}: {QUESTIONS[state.current_question]}"
#         next_btn = "Next Question"
    
#     return transcript, next_question, next_btn, feedback_text

# def start_interview(name):
#     """Initialize the interview with the candidate's name."""
#     if not name.strip():
#         return "Please enter your name to start.", "", "", "", "", gr.update(visible=False), gr.update(visible=True)
    
#     # Reset state for new interview
#     global state
#     state = InterviewState()
#     state.candidate_name = name.strip()
    
#     return (
#         f"Question 1: {QUESTIONS[0]}",  # First question
#         "",                              # Clear transcript
#         "Start Recording",               # Button text
#         "",                              # Clear feedback
#         gr.update(visible=True),         # Show recording controls
#         gr.update(visible=False)         # Hide start section
#     )

# def toggle_recording(recording):
#     """Toggle recording state and update UI accordingly."""
#     state.recording_in_progress = not state.recording_in_progress
#     if recording:
#         return "Stop Recording"
#     return "Start Recording"

# # Create the Gradio interface
# with gr.Blocks(title="AI-Powered Interview Assistant", theme=gr.themes.Soft()) as demo:
#     gr.Markdown("""
#     # 🤖 AI-Powered Interview Assistant
    
#     This assistant will guide you through a series of behavioral interview questions.
#     Your responses will be analyzed for key qualities like humility, learning mindset,
#     and ability to handle feedback.
#     """)
    
#     with gr.Row() as start_section:
#         with gr.Column(scale=1):
#             name_input = gr.Textbox(label="Your Name", placeholder="Enter your full name")
#             start_btn = gr.Button("Start Interview", variant="primary")
    
#     with gr.Row(visible=False) as interview_section:
#         with gr.Column(scale=2):
#             # Current question display
#             question_display = gr.Textbox(
#                 label="Current Question",
#                 interactive=False,
#                 lines=3
#             )
            
#             # Audio input
#             audio_input = gr.Audio(
#                 sources=["microphone"],
#                 type="filepath",
#                 label="Record your answer",
#                 show_download_button=False,
#                 interactive=True
#             )
            
#             # Action buttons
#             with gr.Row():
#                 record_btn = gr.Button("Start Recording", variant="primary")
#                 submit_btn = gr.Button("Submit Answer", interactive=False, variant="secondary")
#                 next_btn = gr.Button("Next Question", visible=False, variant="primary")
            
#             # Transcript display
#             transcript_display = gr.Textbox(
#                 label="Your Response",
#                 placeholder="Your transcribed answer will appear here...",
#                 interactive=False,
#                 lines=4
#             )
            
#             # Feedback display
#             feedback_display = gr.Textbox(
#                 label="Analysis",
#                 placeholder="Analysis of your response will appear here...",
#                 interactive=False,
#                 lines=6
#             )
            
#             # Report section
#             with gr.Group(visible=False) as report_group:
#                 gr.Markdown("## Interview Report")
#                 report_display = gr.HTML()
#                 download_btn = gr.Button("Download Report as PDF")
    
#     # Event handlers
#     start_btn.click(
#         fn=start_interview,
#         inputs=[name_input],
#         outputs=[
#             question_display,
#             transcript_display,
#             record_btn,
#             feedback_display,
#             interview_section,
#             start_section
#         ]
#     )
    
#     # Toggle recording
#     record_btn.click(
#         fn=toggle_recording,
#         inputs=[gr.State(False)],
#         outputs=[record_btn]
#     )
    
#     # Process recorded audio
#     audio_input.stop_recording(
#         fn=process_response,
#         inputs=[audio_input],
#         outputs=[
#             transcript_display,
#             question_display,
#             next_btn,
#             feedback_display
#         ]
#     )
    
#     # Submit button (manual submission)
#     submit_btn.click(
#         fn=process_response,
#         inputs=[audio_input],
#         outputs=[
#             transcript_display,
#             question_display,
#             next_btn,
#             feedback_display
#         ]
#     )
    
#     # Next question
#     next_btn.click(
#         fn=lambda: ("", gr.update(visible=False), gr.update(visible=True)),
#         inputs=None,
#         outputs=[transcript_display, next_btn, submit_btn],
#         show_progress=False
#     )
    
#     # Show/hide report
#     def toggle_report():
#         if state.complete:
#             report_html = generate_interview_report()
#             return gr.update(visible=True), gr.HTML(report_html)
#         return gr.update(visible=False), ""
    
#     next_btn.click(
#         fn=toggle_report,
#         inputs=None,
#         outputs=[report_group, report_display],
#         show_progress=False
#     )
    
#     # Download report
#     def download_report():
#         report_html = generate_interview_report()
#         filename = f"interview_report_{state.candidate_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
#         # Create a directory for reports if it doesn't exist
#         os.makedirs("reports", exist_ok=True)
#         filepath = os.path.join("reports", filename)
        
#         # Generate PDF
#         create_pdf_report(report_html, filepath)
#         return filepath
    
#     download_btn.click(
#         fn=download_report,
#         inputs=None,
#         outputs=gr.File(label="Download Report"),
#         show_progress=False
#     )

# # Run the app
# if __name__ == "__main__":
#     print("Starting the AI-Powered Interview Assistant...")
#     print("Open your browser and navigate to http://localhost:7860")
#     demo.launch(
#         server_name="localhost",
#         server_port=7860,
#         share=False,
#         show_error=True
#     )



































import gradio as gr
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import pyttsx3
from datetime import datetime
import soundfile as sf
# Try to import Whisper model
import os
import sys
import json
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import soundfile as sf
from datetime import datetime

# Try to import Whisper model
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    print("Warning: faster_whisper not found. Using fallback transcription.")
    WHISPER_AVAILABLE = False

# Initialize Whisper model
if WHISPER_AVAILABLE:
    try:
        print("Loading Whisper model (base) on cpu...")
        whisper_model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )
        print("Whisper model loaded successfully!")
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        WHISPER_AVAILABLE = False

sys.path.append(str(Path(__file__).parent))

try:
    from backend.agent_manager import AnalysisResult, run_analysis_pipeline
except ImportError as e:
    print(f"Warning: Could not import backend agents: {e}")
    print("Using fallback analysis functions")
    
    from typing import NamedTuple
    
    class AnalysisResult(NamedTuple):
        """Fallback AnalysisResult class"""
        humility_score: float
        humility_evidence: str
        learning_score: float
        learning_evidence: str
        feedback_score: float
        feedback_evidence: str
        mistakes_score: float
        mistakes_evidence: str
    
    async def run_analysis_pipeline(transcript: str) -> AnalysisResult:
        """Fallback analysis pipeline"""
        return AnalysisResult(
            humility_score=0.0,
            humility_evidence="Analysis not available",
            learning_score=0.0,
            learning_evidence="Analysis not available",
            feedback_score=0.0,
            feedback_evidence="Analysis not available",
            mistakes_score=0.0,
            mistakes_evidence="Analysis not available"
        )

# Interview questions
QUESTIONS = [
    "Can you tell me about a time when you received constructive criticism? How did you handle it?",
    "Describe a situation where you had to work with a difficult team member. How did you handle it?",
    "Tell me about a time when you made a mistake at work. How did you address it?",
    "How do you handle situations where you need to learn something new?",
    "Can you share an example of when you had to adapt to a significant change at work?"
]

class InterviewState:
    def __init__(self):
        self.current_question = 0
        self.answers = {}  # {question_index: {'text': str, 'analysis': dict}}
        self.complete = False
        self.candidate_name = ""
        self.start_time = datetime.now()
        self.scores = {
            'humility': 0,
            'learning': 0,
            'feedback': 0,
            'mistakes': 0
        }
        self.evidence = {}
        self.recording_in_progress = False

# Global state
state = InterviewState()

async def analyze_response(question_index: int, transcript: str) -> dict:
    """Analyze the response using all available agents asynchronously."""
    if not transcript or not transcript.strip():
        return {}
    
    try:
        # Run the analysis pipeline asynchronously
        analysis_result = await run_analysis_pipeline(transcript)
        
        # Convert the analysis result to a dictionary format
        analysis = {
            'humility': {
                'score': analysis_result.humility_score,
                'evidence': analysis_result.humility_evidence
            },
            'learning': {
                'score': analysis_result.learning_score,
                'evidence': analysis_result.learning_evidence
            },
            'feedback': {
                'score': analysis_result.feedback_score,
                'evidence': analysis_result.feedback_evidence
            },
            'mistakes': {
                'score': analysis_result.mistakes_score,
                'evidence': analysis_result.mistakes_evidence
            }
        }
        
        # Update overall scores
        for category in ['humility', 'learning', 'feedback', 'mistakes']:
            state.scores[category] = (state.scores[category] * question_index + analysis[category]['score']) / (question_index + 1)
        
        return analysis
        
    except Exception as e:
        print(f"Error in analyze_response: {e}")
        return {}

def generate_interview_report() -> str:
    """Generate a comprehensive HTML report of the interview."""
    if not state.answers:
        return "<p>No interview data available.</p>"
    
    # Calculate overall score (average of all category scores)
    overall_score = sum(state.scores.values()) / len(state.scores) if state.scores else 0
    
    # Prepare data for the report
    report_data = {
        'candidate_name': state.candidate_name,
        'date': state.start_time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration': str(datetime.now() - state.start_time).split('.')[0],  # Remove microseconds
        'overall_score': overall_score,
        'category_scores': state.scores,
        'questions': [],
        'recommendations': []
    }
    
    # Add question-answer pairs with analysis
    for i, question in enumerate(QUESTIONS[:len(state.answers)]):
        answer_data = state.answers.get(i, {
            'text': 'No answer provided.',
            'analysis': {}
        })
        
        report_data['questions'].append({
            'question': question,
            'answer': answer_data.get('text', 'No answer provided.'),
            'analysis': answer_data.get('analysis', {})
        })
    
    # Generate recommendations based on scores
    if overall_score < 5:
        report_data['recommendations'].append("Consider working on self-awareness and openness to feedback.")
    if state.scores['learning'] < 5:
        report_data['recommendations'].append("Focus on developing a growth mindset and continuous learning.")
    if state.scores['feedback'] < 5:
        report_data['recommendations'].append("Practice seeking and accepting constructive feedback.")
    
    # Generate HTML report
    return generate_html_report(report_data)

def transcribe_audio(audio_path):
    """Transcribe audio data using the Whisper model or fallback to simple processing."""
    if not audio_path or not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return "[No audio file found]"
    
    try:
        # Check if the file exists and has content
        if os.path.getsize(audio_path) == 0:
            print(f"Audio file is empty: {audio_path}")
            return "[Empty audio file]"
            
        print(f"Processing audio file: {audio_path}")
        
        # Read the audio file using soundfile
        try:
            audio_data, sample_rate = sf.read(audio_path)
            print(f"Audio data shape: {audio_data.shape}, Sample rate: {sample_rate}")
        except Exception as e:
            print(f"Error reading audio file with soundfile: {e}")
            # Try with librosa as fallback
            try:
                import librosa
                audio_data, sample_rate = librosa.load(audio_path, sr=16000)
                print(f"Loaded with librosa. Shape: {audio_data.shape}, SR: {sample_rate}")
            except Exception as e2:
                print(f"Error reading audio file with librosa: {e2}")
                return f"[Error reading audio: {str(e2)}]"
        
        # Ensure audio is mono and float32
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=-1)
            
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            
        print(f"Audio data ready for Whisper. Shape: {audio_data.shape}, Type: {audio_data.dtype}")
        
        if WHISPER_AVAILABLE:
            try:
                # Transcribe using Whisper
                segments, info = whisper_model.transcribe(
                    audio_data,
                    language="en",
                    beam_size=5,
                    vad_filter=True
                )
                print("Transcription completed successfully")
                transcription = " ".join(segment.text for segment in segments).strip()
                print(f"Transcription: {transcription}")
                return transcription
            except Exception as e:
                print(f"Whisper transcription failed: {str(e)}")
                return f"[Transcription error: {str(e)}]"
        else:
            print("Whisper not available")
            return "[Audio transcription service not available]"
            
    except Exception as e:
        print(f"Error during audio processing: {str(e)}")
        return f"[Error processing audio: {str(e)}]"

async def process_response(audio_path):
    """Process the recorded response and update the interview state."""
    if state.complete:
        return "", "Interview completed. Please download your report.", "", ""

    print(f"\n{'='*50}\nProcessing response for question {state.current_question + 1}")
    
    # Transcribe the audio
    transcript = transcribe_audio(audio_path)
    
    # Check for errors in transcription
    if not transcript or transcript.startswith("[ERROR]"):
        error_msg = transcript if transcript else "No transcription available"
        print(f"Transcription failed: {error_msg}")
        return "", "Could not transcribe audio. Please try again.", "", ""

    print(f"Transcript: {transcript}")

    # Store the answer
    state.answers[state.current_question] = {
        'text': transcript,
        'analysis': {}
    }
    
    # Analyze the response
    try:
        print("Starting analysis...")
        analysis = await analyze_response(state.current_question, transcript)
        
        if not analysis:
            raise ValueError("Analysis returned empty results")
            
        state.answers[state.current_question]['analysis'] = analysis
        print("Analysis completed successfully")
        
        # Prepare feedback
        feedback = []
        for agent, result in analysis.items():
            feedback.append(f"**{agent.capitalize()}**: {result.get('evidence', 'No specific feedback')}")
        
        feedback_text = "\n\n".join(feedback) if feedback else "No specific feedback available."

    except Exception as e:
        error_msg = f"Error in analysis: {str(e)}"
        print(error_msg)
        feedback_text = "Analysis is currently unavailable. Please try again."

    # Move to next question or complete
    state.current_question += 1
    
    if state.current_question >= len(QUESTIONS):
        state.complete = True
        next_question = "Interview complete! Click 'View Report' to see your results."
        next_btn = "View Report"
    else:
        next_question = f"Question {state.current_question + 1}: {QUESTIONS[state.current_question]}"
        next_btn = "Next Question"
    
    return transcript, next_question, next_btn, feedback_text


def start_interview(name):
    """Initialize the interview with the candidate's name."""
    if not name.strip():
        return "Please enter your name to start.", "", "", "", "", gr.update(visible=False), gr.update(visible=True)
    
    # Reset state for new interview
    global state
    state = InterviewState()
    state.candidate_name = name.strip()
    
    return (
        f"Question 1: {QUESTIONS[0]}",  # First question
        "",                              # Clear transcript
        "Start Recording",               # Button text
        "",                              # Clear feedback
        gr.update(visible=True),         # Show recording controls
        gr.update(visible=False)         # Hide start section
    )

def toggle_recording(recording):
    """Toggle recording state and update UI accordingly."""
    state.recording_in_progress = not state.recording_in_progress
    if recording:
        return "Stop Recording"
    return "Start Recording"

# Create the Gradio interface
with gr.Blocks(title="AI-Powered Interview Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 AI-Powered Interview Assistant
    
    This assistant will guide you through a series of behavioral interview questions.
    Your responses will be analyzed for key qualities like humility, learning mindset,
    and ability to handle feedback.
    """)
    
    with gr.Row(visible=False) as interview_section:
        with gr.Column(scale=2):
            # Current question display
            question_display = gr.Textbox(
                label="Current Question",
                interactive=False,
                lines=3
            )
            
            # Audio input
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",  # This ensures we get a file path
                label="Record your answer",
                show_download_button=False,
                interactive=True,
                format="wav"  # Force WAV format for better compatibility
            )
            
            # Action buttons
            with gr.Row():
                start_btn = gr.Button("Start", variant="primary")
                record_btn = gr.Button("Start Recording", variant="primary")
                submit_btn = gr.Button("Submit Answer", interactive=True, variant="primary")
                next_btn = gr.Button("Next Question", visible=False, variant="primary")
            
            # Transcript display
            transcript_display = gr.Textbox(
                label="Your Response",
                placeholder="Your transcribed answer will appear here...",
                interactive=False,
                lines=4
            )
            
            # Feedback display
            feedback_display = gr.Markdown(
                label="Analysis",
                value="Analysis of your response will appear here...",
                show_label=True
            )
            
            # Report section
            with gr.Group(visible=False) as report_group:
                gr.Markdown("## Interview Report")
                report_display = gr.HTML()
                download_btn = gr.Button("Download Report as PDF")
    
    # Event handlers
    start_btn.click(
        fn=start_interview,
        inputs=[name_input],
        outputs=[
            question_display,
            transcript_display,
            record_btn,
            feedback_display,
            interview_section,
            start_section
        ]
    )
    
    # Toggle recording
    record_btn.click(
        fn=toggle_recording,
        inputs=[gr.State(False)],
        outputs=[record_btn]
    )
    
    # Process recorded audio
    audio_input.stop_recording(
        fn=process_response,
        inputs=[audio_input],
        outputs=[
            transcript_display,
            question_display,
            next_btn,
            feedback_display
        ]
    )
    
    # Submit button (manual submission)
    submit_btn.click(
        fn=process_response,
        inputs=[audio_input],
        outputs=[
            transcript_display,
            question_display,
            next_btn,
            feedback_display
        ]
    )
    
    # Next question
    next_btn.click(
        fn=lambda: ("", gr.update(visible=False), gr.update(visible=True)),
        inputs=None,
        outputs=[transcript_display, next_btn, submit_btn],
        show_progress=False
    )
    
    # Show/hide report
    def toggle_report():
        if state.complete:
            report_html = generate_interview_report()
            return gr.update(visible=True), gr.HTML(report_html)
        return gr.update(visible=False), ""
    
    next_btn.click(
        fn=toggle_report,
        inputs=None,
        outputs=[report_group, report_display],
        show_progress=False
    )
    
    # Download report
    def download_report():
        report_html = generate_interview_report()
        filename = f"interview_report_{state.candidate_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Create a directory for reports if it doesn't exist
        os.makedirs("reports", exist_ok=True)
        filepath = os.path.join("reports", filename)
        
        # Generate PDF
        create_pdf_report(report_html, filepath)
        return filepath
    
    download_btn.click(
        fn=download_report,
        inputs=None,
        outputs=gr.File(label="Download Report"),
        show_progress=False
    )

def validate_audio_file(audio_path: str) -> bool:
    """Validate that the audio file exists and is in a supported format."""
    if not audio_path or not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return False
    
    valid_extensions = ['.wav', '.mp3', '.ogg', '.flac']
    _, ext = os.path.splitext(audio_path.lower())
    
    if ext not in valid_extensions:
        print(f"Unsupported audio format: {ext}")
        return False
    
    return True

def convert_audio_to_wav(audio_path: str) -> str:
    """Convert any audio file to WAV format using ffmpeg if available."""
    if not audio_path or not os.path.exists(audio_path):
        return audio_path
    
    try:
        import subprocess
        import tempfile
        
        # Check if already WAV
        if audio_path.lower().endswith('.wav'):
            return audio_path
            
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            output_path = tmp.name
            
        # Convert using ffmpeg
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if it exists
            '-i', audio_path,
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"Error converting audio: {e.stderr.decode()}")
            return audio_path
            
    except Exception as e:
        print(f"Error in audio conversion: {str(e)}")
        return audio_path

# Run the app
if __name__ == "__main__":
    print("Starting the AI-Powered Interview Assistant...")
    print("Open your browser and navigate to http://localhost:7860")
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True
    )

