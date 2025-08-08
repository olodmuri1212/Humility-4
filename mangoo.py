import gradio as gr
import os
import json
import numpy as np
import torch
import soundfile as sf
import tempfile
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from faster_whisper import WhisperModel
from gtts import gTTS
from io import BytesIO
import asyncio

# Import analysis components
from interview_analyzer import InterviewAnalyzer
from services.report_generator import generate_html_report

# Configuration
MODEL_SIZE = "base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPORTS_DIR = "interview_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Initialize Whisper model
print(f"Loading Whisper model ({MODEL_SIZE}) on {DEVICE}...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8" if DEVICE == "cuda" else "int8")
print("Model loaded successfully!")

# Question Bank
QUESTION_BANK = [
    "Can you tell me about a time when you received constructive criticism? How did you handle it?",
    "Describe a situation where you had to work with a difficult team member. How did you handle it?",
    "Tell me about a time when you made a mistake at work. How did you address it?",
    "How do you handle situations where you need to learn something new?",
    "Can you share an example of when you had to adapt to a significant change at work?"
]

@dataclass
class InterviewState:
    """Tracks the state of the interview."""
    candidate_name: str = ""
    current_question: int = 0
    answers: List[Dict[str, Any]] = field(default_factory=list)
    audio_data: List[Any] = field(default_factory=list)
    transcripts: List[str] = field(default_factory=list)
    analyzer: Optional[InterviewAnalyzer] = None
    complete: bool = False

# Global state
state = InterviewState()

def text_to_speech(text: str, lang: str = 'en') -> tuple:
    """Convert text to speech using gTTS."""
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    audio_data, sample_rate = sf.read(audio_buffer)
    return (sample_rate, audio_data)

async def transcribe_audio(audio_data: tuple) -> str:
    """Transcribe audio using Whisper model."""
    sample_rate, audio_array = audio_data
    
    # Save to temporary file for Whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        sf.write(temp_path, audio_array, sample_rate)
    
    try:
        segments, _ = model.transcribe(
            temp_path,
            language="en",
            beam_size=5,
            vad_filter=True
        )
        return " ".join([segment.text for segment in segments])
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def start_interview(name: str) -> tuple:
    """Initialize the interview with candidate's name."""
    state.candidate_name = name
    state.analyzer = InterviewAnalyzer(candidate_name=name)
    state.current_question = 0
    state.answers = []
    state.audio_data = []
    state.transcripts = [""] * len(QUESTION_BANK)
    state.complete = False
    
    return (
        gr.update(visible=False),  # Name input
        gr.update(visible=True),   # Interview container
        question_display.update(value=get_current_question()),
        gr.update(interactive=True),  # Record button
        gr.update(interactive=False),  # Stop button
        gr.update(value=None),  # Audio output
        gr.update(visible=False),  # Generate transcript button
        gr.update(value=""),  # Transcript display
        gr.update(visible=False),  # Next button
        gr.update(visible=False),  # Finish button
        gr.update(visible=False)   # Report container
    )

def get_current_question() -> str:
    """Get the current question text."""
    if state.current_question < len(QUESTION_BANK):
        return f"Question {state.current_question + 1}/{len(QUESTION_BANK)}: {QUESTION_BANK[state.current_question]}"
    return "Interview Complete"

async def process_audio(audio_data: tuple) -> tuple:
    """Process the recorded audio."""
    if audio_data is None:
        return "No audio detected", False, True, False
    
    # Store audio data for this question
    if len(state.audio_data) <= state.current_question:
        state.audio_data.append(None)
    state.audio_data[state.current_question] = audio_data
    
    return (
        audio_data,  # Audio output
        True,  # Show generate button
        False,  # Disable record button
        False   # Show next/finish buttons
    )

async def generate_transcript() -> tuple:
    """Generate transcript for the current question's audio."""
    if state.current_question >= len(state.audio_data) or state.audio_data[state.current_question] is None:
        return "No audio available to transcribe", False, False, False
    
    # Transcribe the audio
    transcript = await transcribe_audio(state.audio_data[state.current_question])
    state.transcripts[state.current_question] = transcript
    
    # Analyze the response
    question = QUESTION_BANK[state.current_question]
    analysis = await state.analyzer.analyze_response(question, transcript)
    
    # Store the answer and analysis
    if len(state.answers) <= state.current_question:
        state.answers.append({
            "question": question,
            "answer": transcript,
            "analysis": analysis
        })
    else:
        state.answers[state.current_question] = {
            "question": question,
            "answer": transcript,
            "analysis": analysis
        }
    
    # Determine if we should show next or finish button
    is_last_question = state.current_question >= len(QUESTION_BANK) - 1
    
    return (
        transcript,  # Display transcript
        False,  # Hide generate button
        False,  # Keep record button disabled
        True    # Show next/finish buttons
    )

def next_question() -> tuple:
    """Move to the next question."""
    state.current_question += 1
    is_last_question = state.current_question >= len(QUESTION_BANK) - 1
    
    return (
        question_display.update(value=get_current_question()),
        gr.update(interactive=True),  # Enable record button
        gr.update(interactive=False),  # Disable stop button
        gr.update(value=None),  # Clear audio output
        gr.update(visible=False),  # Hide generate button
        gr.update(value=""),  # Clear transcript
        gr.update(visible=False),  # Hide next button
        gr.update(visible=is_last_question),  # Show finish button if last question
        gr.update(visible=False)  # Hide report container
    )

def finish_interview() -> str:
    """Generate and return the final report."""
    state.complete = True
    report_html = generate_html_report(state.analyzer)
    
    # Save the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f"interview_report_{timestamp}.html")
    with open(report_path, "w") as f:
        f.write(report_html)
    
    return report_html

# Gradio UI
with gr.Blocks(title="AI Interview Assistant", theme=gr.themes.Soft()) as demo:
    with gr.Column(visible=True) as name_input:
        gr.Markdown("# 🎙️ AI Interview Assistant")
        name = gr.Textbox(label="Enter your name", placeholder="John Doe")
        start_btn = gr.Button("Start Interview")
    
    with gr.Column(visible=False) as interview_container:
        # Question display
        question_display = gr.Markdown()
        
        # Audio recording
        with gr.Row():
            record_btn = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Record your answer",
                interactive=True
            )
            stop_btn = gr.Button("Stop Recording", interactive=False)
        
        # Audio playback and transcript generation
        audio_output = gr.Audio(label="Recorded Answer", interactive=False, visible=False)
        generate_btn = gr.Button("Generate Transcript", visible=False)
        
        # Transcript display
        transcript_display = gr.Textbox(
            label="Your Answer",
            interactive=False,
            lines=4,
            max_lines=10,
            visible=True
        )
        
        # Navigation buttons
        with gr.Row():
            next_btn = gr.Button("Next Question", visible=False)
            finish_btn = gr.Button("Finish Interview", visible=False)
        
        # Report
        with gr.Column(visible=False) as report_container:
            gr.Markdown("## Interview Report")
            report_html = gr.HTML()
    
    # Event handlers
    start_btn.click(
        fn=start_interview,
        inputs=name,
        outputs=[
            name_input,
            interview_container,
            question_display,
            record_btn,
            stop_btn,
            audio_output,
            generate_btn,
            transcript_display,
            next_btn,
            finish_btn,
            report_container
        ]
    )
    
    record_btn.stop_recording(
        fn=process_audio,
        inputs=record_btn,
        outputs=[
            audio_output,
            generate_btn,
            record_btn,
            stop_btn
        ]
    )
    
    generate_btn.click(
        fn=generate_transcript,
        outputs=[
            transcript_display,
            generate_btn,
            record_btn,
            next_btn
        ]
    )
    
    next_btn.click(
        fn=next_question,
        outputs=[
            question_display,
            record_btn,
            stop_btn,
            audio_output,
            generate_btn,
            transcript_display,
            next_btn,
            finish_btn,
            report_container
        ]
    )
    
    finish_btn.click(
        fn=finish_interview,
        outputs=report_html
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)