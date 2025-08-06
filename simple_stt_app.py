import gradio as gr
import base64
import os
import numpy as np
from faster_whisper import WhisperModel
import torch
import json
from datetime import datetime

# Initialize the Whisper model
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

# Global state
state = InterviewState()

def analyze_response(question, answer):
    """
    Simple analysis of the response.
    In a real application, you would use more sophisticated NLP here.
    """
    # This is a simple keyword-based analysis
    positive_keywords = ["learned", "improved", "grew", "adapt", "collaborat", "team"]
    negative_keywords = ["blame", "fault", "hate", "angry", "useless"]
    
    score = 5  # Base score
    comments = []
    
    answer_lower = answer.lower()
    
    # Check for positive indicators
    for word in positive_keywords:
        if word in answer_lower:
            score += 1
            comments.append(f"Positive indicator: '{word}' shows growth mindset.")
    
    # Check for negative indicators
    for word in negative_keywords:
        if word in answer_lower:
            score -= 1
            comments.append(f"Potential concern: '{word}' might indicate negative attitude.")
    
    # Ensure score is within 1-10 range
    score = max(1, min(10, score))
    
    return score, " ".join(comments) if comments else "No specific feedback available."

def generate_report():
    """Generate a simple text report of the interview."""
    if not state.answers:
        return "No interview data available."
    
    report = []
    report.append("# Interview Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## Responses and Analysis\n")
    
    total_score = 0
    for i, question in enumerate(QUESTIONS[:len(state.answers)]):
        answer = state.answers.get(i, "No answer provided.")
        score = state.scores.get(i, 0)
        comment = state.comments.get(i, "No analysis available.")
        
        report.append(f"### Question {i+1}")
        report.append(f"**Question:** {question}")
        report.append(f"**Answer:** {answer}")
        report.append(f"**Score:** {score}/10")
        report.append(f"**Feedback:** {comment}")
        report.append("")
        
        total_score += score
    
    # Calculate average score
    if state.answers:
        avg_score = total_score / len(state.answers)
        report.append(f"\n## Overall Performance")
        report.append(f"**Average Score:** {avg_score:.1f}/10")
        
        # Simple interpretation
        if avg_score >= 8:
            report.append("**Overall Feedback:** Excellent responses showing strong communication and self-awareness.")
        elif avg_score >= 6:
            report.append("**Overall Feedback:** Good responses with room for improvement in some areas.")
        else:
            report.append("**Overall Feedback:** Some areas need improvement. Consider working on communication and self-reflection.")
    
    return "\n".join(report)

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

def process_response(audio_data):
    """Process the recorded response."""
    if state.complete:
        return "", "Interview already completed. Please download your report.", ""
    
    # Transcribe the audio
    transcript = transcribe_audio(audio_data)
    if not transcript:
        return "", "Could not transcribe audio. Please try again.", ""
    
    # Store the answer
    state.answers[state.current_question] = transcript
    
    # Analyze the response
    score, feedback = analyze_response(QUESTIONS[state.current_question], transcript)
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

# Create the Gradio interface
with gr.Blocks(title="Interview Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ Interview Assistant
    
    This assistant will guide you through a series of interview questions. 
    Record your answers and receive feedback after each response.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Current question display
            question_display = gr.Textbox(
                label="Current Question",
                value=f"Question 1: {QUESTIONS[0]}",
                interactive=False,
                lines=3
            )
            
            # Audio input
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="Record your answer",
                show_download_button=False,
                interactive=True
            )
            
            # Submit button
            submit_btn = gr.Button("Submit Answer", variant="primary")
            
            # Next question button (initially hidden)
            next_btn = gr.Button("Next Question", visible=False)
            
            # Transcript display
            transcript_display = gr.Textbox(
                label="Your Response",
                placeholder="Your transcribed answer will appear here...",
                interactive=False,
                lines=4
            )
            
            # Feedback display
            feedback_display = gr.Textbox(
                label="Feedback",
                placeholder="Feedback will appear here...",
                interactive=False,
                lines=4
            )
            
            # Report section
            with gr.Group(visible=False) as report_group:
                gr.Markdown("## Interview Report")
                report_display = gr.Markdown()
                download_btn = gr.Button("Download Report")
    
    # Event handlers
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
    def toggle_report():
        if state.complete:
            report = generate_report()
            return gr.update(visible=True), gr.Markdown(report)
        return gr.update(visible=False), ""
    
    submit_btn.click(
        fn=toggle_report,
        inputs=None,
        outputs=[report_group, report_display],
        show_progress=False
    )
    
    # Download report
    def download_report():
        report = generate_report()
        filename = f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w') as f:
            f.write(report)
        return filename
    
    download_btn.click(
        fn=download_report,
        inputs=None,
        outputs=gr.File(label="Download Report"),
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
