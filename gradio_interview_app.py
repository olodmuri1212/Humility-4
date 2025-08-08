import gradio as gr
import os
import random
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

# Import analysis components
from interview_analyzer import InterviewAnalyzer

# Setup directories
os.makedirs("recordings", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Question Bank
QUESTION_BANK = [
    "Can you tell me about a time when you received constructive criticism?",
    "Describe a situation where you had to work with a difficult team member.",
    "Tell me about a time when you made a mistake at work.",
    "How do you handle situations where you need to learn something new?",
    "Can you share an example of when you had to adapt to a significant change?",
    "Describe a time when you had to give someone difficult feedback.",
    "Tell me about a time when you had to work with someone with a different working style.",
    "How do you handle situations where you don't know the answer to something?",
    "Describe a time when you had to learn a new skill quickly.",
    "Tell me about a time when you had to admit you were wrong."
]

class InterviewState:
    """Manages the interview state and data."""
    def __init__(self):
        self.candidate_name = ""
        self.current_question = 0
        self.questions: List[str] = []
        self.answers: Dict[int, str] = {}
        self.audio_files: Dict[int, str] = {}
        self.transcripts: Dict[int, str] = {}
        self.analysis: Dict[int, dict] = {}
        self.analyzer = None
        self.start_time = None

    def initialize(self, name: str, num_questions: int = 5):
        """Initialize a new interview session."""
        self.candidate_name = name
        self.questions = random.sample(QUESTION_BANK, min(num_questions, len(QUESTION_BANK)))
        self.current_question = 0
        self.answers = {}
        self.audio_files = {}
        self.transcripts = {}
        self.analysis = {}
        self.analyzer = InterviewAnalyzer(candidate_name=name)
        self.start_time = datetime.now()

    def save_audio(self, audio_path: str):
        """Save the recorded audio for the current question."""
        if not os.path.exists(audio_path):
            return False
            
        save_path = f"recordings/{self.candidate_name}_q{self.current_question+1}.wav"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # In a real app, you would copy/process the audio file here
        self.audio_files[self.current_question] = save_path
        return True

    async def process_response(self, audio_path: str) -> tuple[str, dict]:
        """Process the recorded response."""
        if not os.path.exists(audio_path):
            return "Error: Audio file not found.", {}
        
        # Save audio
        self.save_audio(audio_path)
        
        # Transcribe (placeholder - implement actual STT)
        transcript = await self.transcribe_audio(audio_path)
        self.transcripts[self.current_question] = transcript
        
        # Analyze response
        analysis = await self.analyzer.analyze_response(
            self.questions[self.current_question],
            transcript
        )
        self.analysis[self.current_question] = analysis
        
        return transcript, analysis
    
    async def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper (placeholder)."""
        # In a real implementation, this would call the Whisper model
        return f"[Transcript of answer to: {self.questions[self.current_question]}]"

    def generate_report(self) -> str:
        """Generate an HTML report of the interview."""
        if not self.analyzer:
            return "<h2>Error: Interview not started</h2>"
        
        try:
            # Generate report using the analyzer
            report = self.analyzer.generate_report()
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"reports/{self.candidate_name}_{timestamp}.html"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
                
            return report
            
        except Exception as e:
            return f"<h2>Error generating report: {str(e)}</h2>"

# Global state
state = InterviewState()

# Create the Gradio interface
def create_ui():
    with gr.Blocks(title="Automated Interview System") as demo:
        # State
        state_var = gr.State()
        
        # UI Components
        with gr.Row():
            # Left Panel
            with gr.Column(scale=1):
                # Start Screen
                with gr.Group() as start_group:
                    name_input = gr.Textbox(label="Your Name")
                    start_btn = gr.Button("Start Interview")
                
                # Question Interface
                with gr.Group(visible=False) as question_group:
                    question_display = gr.Markdown()
                    audio_input = gr.Audio(type="filepath")
                    record_btn = gr.Button("Record Answer")
                    
                    with gr.Row():
                        prev_btn = gr.Button("Previous")
                        next_btn = gr.Button("Next")
                
                # Progress
                with gr.Group(visible=False) as progress_group:
                    progress_text = gr.Markdown("Question 1 of 5")
                    progress_bar = gr.Slider(0, 5, 0, interactive=False)
            
            # Right Panel
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("Current Response"):
                        transcript_display = gr.Textbox(label="Transcript", interactive=False)
                        analysis_display = gr.JSON(label="Analysis")
                    
                    with gr.TabItem("All Responses"):
                        responses_table = gr.Dataframe(
                            headers=["Question", "Response"],
                            datatype=["str", "str"],
                            interactive=False
                        )
                
                with gr.Group(visible=False) as report_group:
                    report_display = gr.HTML()
                    download_btn = gr.Button("Download Report")
        
        # Event Handlers
        def start_interview(name):
            if not name.strip():
                raise gr.Error("Please enter your name")
            
            state.initialize(name.strip())
            
            return [
                gr.update(visible=True),  # question_group
                gr.update(visible=True),  # progress_group
                gr.update(visible=False),  # start_group
                gr.update(value=state.questions[0]),  # question_display
                gr.update(value=1, maximum=len(state.questions)),  # progress_bar
                "",  # Clear transcript
                {},  # Clear analysis
                state
            ]
        
        async def process_response(audio_file, current_state):
            if not audio_file:
                raise gr.Error("Please record your answer first")
            
            # Process the audio and get transcript
            transcript, analysis = await state.process_response(audio_file)
            
            # Move to next question if not the last one
            if state.current_question < len(state.questions) - 1:
                state.current_question += 1
            
            # Check if we should show the report (last question answered)
            show_report = state.current_question >= len(state.questions) - 1 and \
                        len(state.transcripts) == len(state.questions)
            
            return [
                gr.update(value=state.questions[state.current_question]),  # Update question
                gr.update(value=state.current_question + 1),  # Update progress
                transcript,  # Show transcript
                analysis,  # Show analysis
                gr.update(visible=show_report),  # Show report if last question
                state
            ]
        
        def navigate_question(direction):
            """Handle Previous/Next navigation."""
            if direction == "next" and state.current_question < len(state.questions) - 1:
                state.current_question += 1
            elif direction == "prev" and state.current_question > 0:
                state.current_question -= 1
            
            # Get current question data
            current_q = state.questions[state.current_question]
            current_transcript = state.transcripts.get(state.current_question, "")
            current_analysis = state.analysis.get(state.current_question, {})
            
            return [
                gr.update(value=current_q),  # Update question display
                gr.update(value=state.current_question + 1),  # Update progress
                current_transcript,  # Update transcript
                current_analysis,  # Update analysis
                gr.update(visible=False)  # Hide report when navigating
            ]
        
        # Connect UI components
        start_btn.click(
            fn=start_interview,
            inputs=[name_input],
            outputs=[
                question_group, progress_group, start_group,
                question_display, progress_bar,
                transcript_display, analysis_display,
                state_var
            ]
        )
        
        record_btn.click(
            fn=process_response,
            inputs=[audio_input, state_var],
            outputs=[
                question_display, progress_bar,
                transcript_display, analysis_display,
                report_group,
                state_var
            ]
        )
        
        next_btn.click(
            fn=lambda: navigate_question("next"),
            inputs=None,
            outputs=[
                question_display, progress_bar,
                transcript_display, analysis_display,
                report_group
            ]
        )
        
        prev_btn.click(
            fn=lambda: navigate_question("prev"),
            inputs=None,
            outputs=[
                question_display, progress_bar,
                transcript_display, analysis_display,
                report_group
            ]
        )
        
        return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
