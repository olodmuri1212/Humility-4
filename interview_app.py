import gradio as gr
import base64
import os
import numpy as np
from faster_whisper import WhisperModel
import torch
import json
from datetime import datetime
from gtts import gTTS
from io import BytesIO
import soundfile as sf
import asyncio
from interview_analyzer import InterviewAnalyzer

# Ensure the reports directory exists
os.makedirs("reports", exist_ok=True)

# Initialize the Whisper model for speech-to-text
MODEL_SIZE = "base"  # You can change this to "small", "medium", or "large" for better accuracy
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading Whisper model ({MODEL_SIZE}) on {DEVICE}...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8" if DEVICE == "cuda" else "int8")
print("Model loaded successfully!")

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
        self.answers = {}
        self.scores = {}
        self.comments = {}
        self.complete = False
        self.candidate_name = ""
        self.conversation_history = []

# Global state
state = InterviewState()

def text_to_speech(text, lang='en'):
    """Convert text to speech using gTTS and return audio data"""
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    
    # Convert to numpy array for Gradio
    audio_data, sample_rate = sf.read(audio_buffer)
    return (sample_rate, audio_data)

async def analyze_response(question, answer):
    """Analyze the response using the InterviewAnalyzer."""
    if not hasattr(state, 'analyzer'):
        state.analyzer = InterviewAnalyzer(candidate_name=getattr(state, 'candidate_name', 'Candidate'))
    
    # Get analysis from the InterviewAnalyzer
    analysis = await state.analyzer.analyze_response(question, answer)
    
    if "error" in analysis:
        return 5, "Analysis unavailable at this time."
    
    # Format the feedback
    score = analysis["overall_score"]
    feedback_parts = []
    
    for trait, evidence in analysis["evidence"].items():
        if evidence and evidence != "No specific evidence found.":
            feedback_parts.append(f"{trait.capitalize()}: {evidence}")
    
    # Add suggestions if any
    if analysis.get("suggestions"):
        feedback_parts.append("\nSuggestions: " + "; ".join(analysis["suggestions"]))
    
    feedback = "\n".join(feedback_parts)
    return score, feedback if feedback else "No specific feedback available."

async def generate_report():
    """Generate a comprehensive report using the InterviewAnalyzer."""
    if not hasattr(state, 'analyzer') or not state.answers:
        return "No interview data available."
    
    # Generate the HTML report
    html_report = state.analyzer.generate_report(format="html")
    
    # Save the report to a file
    report_path = state.analyzer.save_report(directory="reports", format="html")
    
    # Also save as PDF if needed
    pdf_path = report_path.replace(".html", ".pdf")
    try:
        from weasyprint import HTML
        HTML(string=html_report).write_pdf(pdf_path)
    except Exception as e:
        print(f"Could not generate PDF: {e}")
    
    return html_report, report_path, pdf_path

def transcribe_audio(audio_data):
    """
    Transcribe audio data using the Whisper model.
    """
    if audio_data is None:
        return ""
        
    try:
        # Handle different audio data formats
        if isinstance(audio_data, tuple) and len(audio_data) == 2:
            sample_rate, audio_array = audio_data
        else:
            audio_array = audio_data
            sample_rate = 16000
            
        # Ensure audio is in the correct format (float32, mono)
        if hasattr(audio_array, 'shape') and len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1) if audio_array.shape[1] == 2 else audio_array
            
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32) / np.iinfo(audio_array.dtype).max
            
        # Resample if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
        # Transcribe the audio
        segments, _ = model.transcribe(
            audio_array,
            language="en",
            beam_size=5,
            vad_filter=True
        )
        
        # Combine all segments into a single transcript
        transcript = " ".join([segment.text for segment in segments])
        return transcript.strip()
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return ""

async def start_interview(name):
    """Initialize the interview with candidate's name"""
    if not name or name.strip() == "":
        return "Please enter your name.", "", None, gr.update(visible=True), gr.update(visible=True)
    
    state.candidate_name = name.strip()
    state.current_question = 0
    state.answers = {}
    state.scores = {}
    state.comments = {}
    state.complete = False
    state.conversation_history = []
    
    # Generate speech for the first question
    question = QUESTIONS[0]
    audio = text_to_speech(question)
    
    return f"Question 1: {question}", "", audio, gr.update(visible=True), gr.update(visible=True)

async def process_response(audio_data):
    """Process the recorded response."""
    if state.complete:
        return "", "Interview already completed. Please download your report.", ""
    
    # Transcribe the audio
    transcript = transcribe_audio(audio_data)
    if not transcript:
        return "", "Could not transcribe audio. Please try again.", ""
    
    # Store the answer
    state.answers[state.current_question] = transcript
    
    # Analyze the response asynchronously
    score, feedback = await analyze_response(QUESTIONS[state.current_question], transcript)
    state.scores[state.current_question] = score
    state.comments[state.current_question] = feedback
    
    # Move to next question or complete
    state.current_question += 1
    
    if state.current_question >= len(QUESTIONS):
        state.complete = True
        next_question = "Interview complete! Click 'Generate Report' to view your results."
        next_btn = "Generate Report"
    else:
        next_question = f"Question {state.current_question + 1}: {QUESTIONS[state.current_question]}"
        next_btn = "Next Question"
    
    return transcript, next_question, next_btn

def get_next_question():
    """Get the next question in the interview."""
    if state.complete:
        return "Interview complete! Click 'Generate Report' to view your results.", None, gr.update(visible=False), gr.update(visible=True)
    
    question = QUESTIONS[state.current_question]
    audio = text_to_speech(question)
    
    return f"Question {state.current_question + 1}: {question}", audio, gr.update(visible=True), gr.update(visible=False)

# Create the Gradio interface
with gr.Blocks(title="Interview Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ Interview Assistant
    
    This assistant will guide you through a series of interview questions. 
    Listen to each question, record your answer, and receive feedback.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Candidate name input
            name_input = gr.Textbox(
                label="Enter your name",
                placeholder="John Doe",
                interactive=True
            )
            start_btn = gr.Button("Start Interview", variant="primary")
            
            # Question display
            question_display = gr.Textbox(
                label="Current Question",
                placeholder="Click 'Start Interview' to begin...",
                interactive=False,
                lines=3
            )
            
            # Audio output for question
            audio_output = gr.Audio(
                label="Listen to the question",
                type="numpy",
                interactive=False,
                visible=False
            )
            
            # Audio input for answer
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Record your answer",
                show_download_button=False,
                interactive=True,
                visible=False
            )
            
            # Submit button
            submit_btn = gr.Button("Submit Answer", variant="primary", visible=False)
            
            # Next question button
            next_btn = gr.Button("Next Question", visible=False)
            
            # Transcript display
            transcript_display = gr.Textbox(
                label="Your Response",
                placeholder="Your transcribed answer will appear here...",
                interactive=False,
                lines=4
            )
            
            # Report section
            with gr.Group(visible=False) as report_group:
                gr.Markdown("## Interview Report")
                report_display = gr.HTML()
                with gr.Row():
                    download_html_btn = gr.Button("Download HTML Report")
                    download_pdf_btn = gr.Button("Download PDF Report")
    
    # Event handlers
    start_btn.click(
        fn=start_interview,
        inputs=[name_input],
        outputs=[question_display, transcript_display, audio_output, audio_input, submit_btn],
        show_progress=False
    )
    
    submit_btn.click(
        fn=process_response,
        inputs=[audio_input],
        outputs=[transcript_display, question_display, submit_btn],
        show_progress=False
    )
    
    next_btn.click(
        fn=lambda: ("", gr.update(visible=False), gr.update(visible=True)),
        inputs=None,
        outputs=[transcript_display, next_btn, submit_btn],
        show_progress=False
    )
    
    # Show/hide report
    async def toggle_report():
        if state.complete:
            html_report, report_path, pdf_path = await generate_report()
            return (
                gr.update(visible=True), 
                gr.HTML(html_report),
                gr.File(report_path),
                gr.File(pdf_path) if os.path.exists(pdf_path) else None
            )
        return gr.update(visible=False), "", None, None
    
    submit_btn.click(
        fn=toggle_report,
        inputs=None,
        outputs=[report_group, report_display, download_html_btn, download_pdf_btn],
        show_progress=False
    )
    
    # Download report handlers
    def get_report_path(ext):
        if not hasattr(state, 'analyzer'):
            return None
        safe_name = "".join(c if c.isalnum() else "_" for c in getattr(state, 'candidate_name', 'candidate'))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"interview_report_{safe_name}_{timestamp}.{ext}"
        return os.path.join("reports", filename)
    
    def download_html_report():
        if not hasattr(state, 'analyzer'):
            return None
        report_path = get_report_path("html")
        if report_path and os.path.exists(report_path):
            return report_path
        return None
    
    def download_pdf_report():
        if not hasattr(state, 'analyzer'):
            return None
        pdf_path = get_report_path("pdf")
        if pdf_path and os.path.exists(pdf_path):
            return pdf_path
        return None
    
    download_html_btn.click(
        fn=download_html_report,
        inputs=None,
        outputs=gr.File(label="Download HTML Report"),
        show_progress=False
    )
    
    download_pdf_btn.click(
        fn=download_pdf_report,
        inputs=None,
        outputs=gr.File(label="Download PDF Report"),
        show_progress=False
    )

# Run the app
if __name__ == "__main__":
    print("Starting the Interview Assistant...")
    print("Open your browser and navigate to http://localhost:7860")
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True
    )
