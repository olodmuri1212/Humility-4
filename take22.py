import gradio as gr
import requests
import base64
import json
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os

# Configuration
BACKEND_URL = "http://localhost:8000"  # Update if backend runs on a different URL
SAMPLE_RATE = 24000
QUESTION_BANK = [
    "Tell me about a time you were wrong.",
    "Describe a situation where you received constructive feedback. How did you respond?",
    "Can you share an example of when you had to admit you didn't know something?",
    "Tell me about a time you made a mistake at work. How did you handle it?",
    "Describe a time when someone you managed was better than you at something. How did you handle that?"
]

class InterviewState:
    """Maintains the state of the interview session."""
    def __init__(self):
        self.current_question_idx = 0
        self.audio_data = [None] * len(QUESTION_BANK)
        self.transcripts = [""] * len(QUESTION_BANK)
        self.analyses = [None] * len(QUESTION_BANK)
        self.candidate_name = ""
        self.interview_id = str(int(time.time()))

# Global state
state = InterviewState()

def transcribe_audio(audio_path: str) -> str:
    """Send audio to backend for transcription."""
    try:
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            
        response = requests.post(
            f"{BACKEND_URL}/transcribe",
            files={"file": ("audio.wav", audio_bytes, "audio/wav")}
        )
        
        if response.status_code == 200:
            return response.json().get("transcript", "")
        else:
            print(f"Transcription failed: {response.text}")
            return ""
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return ""

def analyze_response(question: str, transcript: str) -> Optional[Dict]:
    """Send transcript to backend for analysis."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/analyze",
            json={"transcript": transcript, "question": question},
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Analysis failed: {response.text}")
            return None
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return None

def save_audio(audio_path: str) -> Tuple[str, str, str]:
    """Save recorded audio and process it."""
    if state.current_question_idx >= len(QUESTION_BANK):
        return "", "Interview complete!", ""
    
    # Save audio data
    state.audio_data[state.current_question_idx] = audio_path
    
    # Transcribe audio
    transcript = transcribe_audio(audio_path)
    state.transcripts[state.current_question_idx] = transcript
    
    # Analyze response
    analysis = analyze_response(
        QUESTION_BANK[state.current_question_idx], 
        transcript
    )
    state.analyses[state.current_question_idx] = analysis
    
    # Move to next question
    state.current_question_idx += 1
    
    # Prepare next question or finish
    if state.current_question_idx < len(QUESTION_BANK):
        next_question = f"Question {state.current_question_idx + 1}: {QUESTION_BANK[state.current_question_idx]}"
        next_btn = "Next Question"
    else:
        next_question = "Interview complete! Click 'Generate Report' to view your results."
        next_btn = "Generate Report"
    
    return transcript, next_question, next_btn

def generate_report() -> str:
    """Generate a report from all interview responses."""
    report = "# Interview Report\n\n"
    report += f"## Candidate: {state.candidate_name or 'Not specified'}\n\n"
    
    for i, (question, transcript, analysis) in enumerate(zip(QUESTION_BANK, state.transcripts, state.analyses)):
        report += f"### Question {i+1}: {question}\n"
        report += f"**Response:** {transcript or 'No response recorded'}\n\n"
        
        if analysis and 'scores' in analysis:
            report += "**Analysis:**\n"
            for agent, details in analysis['scores'].items():
                report += f"- {agent}: {details.get('score', 'N/A')} - {details.get('evidence', 'No evidence')}\n"
        report += "\n---\n\n"
    
    return report

# Gradio UI Components
def create_ui():
    with gr.Blocks(title="Humility Interview Assistant") as demo:
        gr.Markdown("# Humility Interview Assistant")
        
        with gr.Row():
            with gr.Column(scale=1):
                name_input = gr.Textbox(label="Candidate Name", placeholder="Enter candidate name")
                question_display = gr.Markdown("## " + QUESTION_BANK[0])
                audio_input = gr.Audio(source="microphone", type="filepath", label="Record your answer")
                submit_btn = gr.Button("Submit Answer")
                next_btn = gr.Button("Next Question", visible=True)
                
                with gr.Row():
                    prev_btn = gr.Button("Previous Question")
                    report_btn = gr.Button("Generate Report", visible=False)
                
                status = gr.Markdown("")
                
            with gr.Column(scale=1):
                transcript_display = gr.Textbox(label="Transcript", lines=10, interactive=False)
                analysis_display = gr.Markdown("## Analysis will appear here")
                report_display = gr.Markdown("", visible=False)
        
        # Event handlers
        def update_name(name):
            state.candidate_name = name
            return name
            
        def on_submit(audio_path):
            if not audio_path:
                return "", "Please record an answer before submitting.", ""
            return save_audio(audio_path)
            
        def on_next():
            if state.current_question_idx < len(QUESTION_BANK):
                return {
                    question_display: gr.Markdown.update(value=f"## {QUESTION_BANK[state.current_question_idx]}"),
                    next_btn: gr.Button.update(
                        visible=state.current_question_idx < len(QUESTION_BANK) - 1
                    ),
                    report_btn: gr.Button.update(
                        visible=state.current_question_idx >= len(QUESTION_BANK) - 1
                    )
                }
            return {}
            
        def on_previous():
            if state.current_question_idx > 0:
                state.current_question_idx -= 1
                return {
                    question_display: gr.Markdown.update(value=f"## {QUESTION_BANK[state.current_question_idx]}"),
                    transcript_display: state.transcripts[state.current_question_idx] or "",
                    next_btn: gr.Button.update(visible=True),
                    report_btn: gr.Button.update(visible=False)
                }
            return {}
            
        def show_report():
            return {
                report_display: gr.Markdown.update(
                    value=generate_report(),
                    visible=True
                )
            }
        
        # Connect UI components to handlers
        name_input.change(fn=update_name, inputs=name_input, outputs=name_input)
        submit_btn.click(
            fn=on_submit,
            inputs=[audio_input],
            outputs=[transcript_display, status, next_btn]
        )
        next_btn.click(
            fn=on_next,
            inputs=[],
            outputs=[question_display, next_btn, report_btn]
        )
        prev_btn.click(
            fn=on_previous,
            inputs=[],
            outputs=[question_display, transcript_display, next_btn, report_btn]
        )
        report_btn.click(
            fn=show_report,
            inputs=[],
            outputs=[report_display]
        )
        
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
