# import gradio as gr
# import requests
# import base64
# import json
# import time
# from typing import List, Dict, Optional, Tuple
# from pathlib import Path
# import os

# # Configuration
# BACKEND_URL = "http://localhost:8000"  # Update if backend runs on a different URL
# SAMPLE_RATE = 24000
# QUESTION_BANK = [
#     "Tell me about a time you were wrong.",
#     "Describe a situation where you received constructive feedback. How did you respond?",
#     "Can you share an example of when you had to admit you didn't know something?",
#     "Tell me about a time you made a mistake at work. How did you handle it?",
#     "Describe a time when someone you managed was better than you at something. How did you handle that?"
# ]

# class InterviewState:
#     """Maintains the state of the interview session."""
#     def __init__(self):
#         self.current_question_idx = 0
#         self.audio_data = [None] * len(QUESTION_BANK)
#         self.transcripts = [""] * len(QUESTION_BANK)
#         self.analyses = [None] * len(QUESTION_BANK)
#         self.candidate_name = ""
#         self.interview_id = str(int(time.time()))

# # Global state
# state = InterviewState()

# def transcribe_audio(audio_path: str) -> str:
#     """Send audio to backend for transcription."""
#     try:
#         with open(audio_path, "rb") as audio_file:
#             audio_bytes = audio_file.read()
            
#         response = requests.post(
#             f"{BACKEND_URL}/transcribe",
#             files={"file": ("audio.wav", audio_bytes, "audio/wav")}
#         )
        
#         if response.status_code == 200:
#             return response.json().get("transcript", "")
#         else:
#             print(f"Transcription failed: {response.text}")
#             return ""
#     except Exception as e:
#         print(f"Error in transcription: {str(e)}")
#         return ""

# def analyze_response(question: str, transcript: str) -> Optional[Dict]:
#     """Send transcript to backend for analysis."""
#     try:
#         response = requests.post(
#             f"{BACKEND_URL}/analyze",
#             json={"transcript": transcript, "question": question},
#             timeout=60
#         )
#         if response.status_code == 200:
#             return response.json()
#         else:
#             print(f"Analysis failed: {response.text}")
#             return None
#     except Exception as e:
#         print(f"Error in analysis: {str(e)}")
#         return None

# def save_audio(audio_path: str) -> Tuple[str, str, str]:
#     """Save recorded audio and process it."""
#     if state.current_question_idx >= len(QUESTION_BANK):
#         return "", "Interview complete!", ""
    
#     # Save audio data
#     state.audio_data[state.current_question_idx] = audio_path
    
#     # Transcribe audio
#     transcript = transcribe_audio(audio_path)
#     state.transcripts[state.current_question_idx] = transcript
    
#     # Analyze response
#     analysis = analyze_response(
#         QUESTION_BANK[state.current_question_idx], 
#         transcript
#     )
#     state.analyses[state.current_question_idx] = analysis
    
#     # Move to next question
#     state.current_question_idx += 1
    
#     # Prepare next question or finish
#     if state.current_question_idx < len(QUESTION_BANK):
#         next_question = f"Question {state.current_question_idx + 1}: {QUESTION_BANK[state.current_question_idx]}"
#         next_btn = "Next Question"
#     else:
#         next_question = "Interview complete! Click 'Generate Report' to view your results."
#         next_btn = "Generate Report"
    
#     return transcript, next_question, next_btn

# def generate_report() -> str:
#     """Generate a report from all interview responses."""
#     report = "# Interview Report\n\n"
#     report += f"## Candidate: {state.candidate_name or 'Not specified'}\n\n"
    
#     for i, (question, transcript, analysis) in enumerate(zip(QUESTION_BANK, state.transcripts, state.analyses)):
#         report += f"### Question {i+1}: {question}\n"
#         report += f"**Response:** {transcript or 'No response recorded'}\n\n"
        
#         if analysis and 'scores' in analysis:
#             report += "**Analysis:**\n"
#             for agent, details in analysis['scores'].items():
#                 report += f"- {agent}: {details.get('score', 'N/A')} - {details.get('evidence', 'No evidence')}\n"
#         report += "\n---\n\n"
    
#     return report

# # Gradio UI Components
# def create_ui():
#     with gr.Blocks(title="Humility Interview Assistant") as demo:
#         gr.Markdown("# Humility Interview Assistant")
        
#         with gr.Row():
#             with gr.Column(scale=1):
#                 name_input = gr.Textbox(label="Candidate Name", placeholder="Enter candidate name")
#                 question_display = gr.Markdown("## " + QUESTION_BANK[0])
#                 audio_input = gr.Audio(source="microphone", type="filepath", label="Record your answer")
#                 submit_btn = gr.Button("Submit Answer")
#                 next_btn = gr.Button("Next Question", visible=True)
                
#                 with gr.Row():
#                     prev_btn = gr.Button("Previous Question")
#                     report_btn = gr.Button("Generate Report", visible=False)
                
#                 status = gr.Markdown("")
                
#             with gr.Column(scale=1):
#                 transcript_display = gr.Textbox(label="Transcript", lines=10, interactive=False)
#                 analysis_display = gr.Markdown("## Analysis will appear here")
#                 report_display = gr.Markdown("", visible=False)
        
#         # Event handlers
#         def update_name(name):
#             state.candidate_name = name
#             return name
            
#         def on_submit(audio_path):
#             if not audio_path:
#                 return "", "Please record an answer before submitting.", ""
#             return save_audio(audio_path)
            
#         def on_next():
#             if state.current_question_idx < len(QUESTION_BANK):
#                 return {
#                     question_display: gr.Markdown.update(value=f"## {QUESTION_BANK[state.current_question_idx]}"),
#                     next_btn: gr.Button.update(
#                         visible=state.current_question_idx < len(QUESTION_BANK) - 1
#                     ),
#                     report_btn: gr.Button.update(
#                         visible=state.current_question_idx >= len(QUESTION_BANK) - 1
#                     )
#                 }
#             return {}
            
#         def on_previous():
#             if state.current_question_idx > 0:
#                 state.current_question_idx -= 1
#                 return {
#                     question_display: gr.Markdown.update(value=f"## {QUESTION_BANK[state.current_question_idx]}"),
#                     transcript_display: state.transcripts[state.current_question_idx] or "",
#                     next_btn: gr.Button.update(visible=True),
#                     report_btn: gr.Button.update(visible=False)
#                 }
#             return {}
            
#         def show_report():
#             return {
#                 report_display: gr.Markdown.update(
#                     value=generate_report(),
#                     visible=True
#                 )
#             }
        
#         # Connect UI components to handlers
#         name_input.change(fn=update_name, inputs=name_input, outputs=name_input)
#         submit_btn.click(
#             fn=on_submit,
#             inputs=[audio_input],
#             outputs=[transcript_display, status, next_btn]
#         )
#         next_btn.click(
#             fn=on_next,
#             inputs=[],
#             outputs=[question_display, next_btn, report_btn]
#         )
#         prev_btn.click(
#             fn=on_previous,
#             inputs=[],
#             outputs=[question_display, transcript_display, next_btn, report_btn]
#         )
#         report_btn.click(
#             fn=show_report,
#             inputs=[],
#             outputs=[report_display]
#         )
        
#     return demo

# if __name__ == "__main__":
#     demo = create_ui()
#     demo.launch(server_name="0.0.0.0", server_port=7860)























# import gradio as gr
# from datetime import datetime
# import soundfile as sf
# import asyncio
# from backend.agent_manager import run_analysis_pipeline

# # Define the questions and their audio files
# questions = [
#     "Can you tell me about a time when you received constructive criticism?",
#     "Describe a situation where you had to work with a difficult team member.",
#     "Tell me about a time when you made a mistake at work.",
#     "How do you handle situations where you need to learn something new?",
#     "Can you share an example of when you had to adapt to a significant change at work?"
# ]

# # Placeholder for audio files (replace with actual paths)
# audio_files = [
#     "audio/question1.mp3",
#     "audio/question2.mp3",
#     "audio/question3.mp3",
#     "audio/question4.mp3",
#     "audio/question5.mp3"
# ]

# # Initialize the question index in the global scope
# current_question_index = 0

# with gr.Blocks(title="Interview Assistant") as demo:
#     gr.Markdown("## Interview Assistant")
    
#     candidate_name = gr.Textbox(
#         label="Candidate Name",
#         placeholder="Enter candidate name"
#     )

#     question_display = gr.Markdown(f"**Question 1:** {questions[0]}")
#     audio_player = gr.Audio(
#         audio_files[0],
#         label="Listen to the question",
#         type="filepath"
#     )
#     answer_box = gr.Audio(
#         label="Record your answer",
#         type="numpy",
#         interactive=True
#     )

#     analyze_button = gr.Button("Analyze Answer")
#     results_output = gr.Textbox(
#         label="Analysis Results",
#         interactive=False
#     )

#     async def on_analyze_answer(name, audio_data):
#         global current_question_index  # Use global instead of nonlocal since it's in global scope
#         transcript = transcribe_audio(audio_data)
#         detailed_scores, overall_score = await analyze_answers(transcript)
#         result_text = format_results(detailed_scores, overall_score)

#         # Update the question index for the next question
#         current_question_index += 1

#         # Check if there are more questions
#         if current_question_index < len(questions):
#             question_display.update(
#                 value=f"**Question {current_question_index + 1}:** {questions[current_question_index]}"
#             )
#             audio_player.update(
#                 value=audio_files[current_question_index]
#             )

#         return result_text

#     analyze_button.click(
#         fn=on_analyze_answer,
#         inputs=[candidate_name, answer_box],
#         outputs=results_output
#     )

# if __name__ == "__main__":
#     demo.launch(server_name="localhost", server_port=7860)















# interview_app.py
import os
import numpy as np
import torch
import gradio as gr
import librosa
import asyncio
from faster_whisper import WhisperModel
from datetime import datetime
from io import BytesIO
import soundfile as sf

# Your existing modules (make sure these import paths match your repo)
from interview_state import SessionState, Turn
from interview_analyzer import InterviewAnalyzer
from backend.agent_manager import run_analysis_pipeline  # returns list of dicts [{'agent_name':..., 'score':...}, ...]
import report_double_generate

# Make sure reports dir exists
os.makedirs("reports", exist_ok=True)

# Questions (same as before)
QUESTIONS = [
    "1) Can you tell me about a time when you received constructive criticism? How did you handle it?",
    "2) Describe a situation where you had to work with a difficult team member. How did you handle it?",
    "3) Tell me about a time when you made a mistake at work. How did you address it?",
    "4) How do you handle situations where you need to learn something new?",
    "5) Can you share an example of when you had to adapt to a significant change at work?"
]

# Whisper model setup
MODEL_SIZE = "base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8" if DEVICE == "cuda" else "int8")

# Session/State
session = SessionState()  # re-use your existing SessionState dataclass
analyzer: InterviewAnalyzer | None = None

# --- Helpers ---

def preprocess_audio(audio_data):
    """Convert gr.Audio numpy → float32 mono @16kHz."""
    if audio_data is None:
        return np.zeros((0,), dtype=np.float32)

    sr, arr = audio_data
    if arr is None:
        return np.zeros((0,), dtype=np.float32)

    # if stereo, convert to mono
    if arr.ndim > 1 and arr.shape[1] == 2:
        arr = arr.mean(axis=1)

    # normalize to float32
    if arr.dtype != np.float32:
        # if integer type, convert to float32
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        else:
            arr = arr.astype(np.float32)

    if sr != 16000:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)

    return arr

async def transcribe(audio_data):
    """Run Whisper on raw audio asynchronously."""
    audio = preprocess_audio(audio_data)
    try:
        segments, _ = whisper_model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True
        )
        return " ".join(seg.text for seg in segments).strip()
    except Exception as e:
        print(f"[transcribe] error: {e}")
        return ""

async def run_cumulative_agent_analysis(answers_list):
    """
    Send concatenated answers to `run_analysis_pipeline` and return structured breakdown:
    returns tuple (per_agent_list, overall_avg)
    per_agent_list: list of dicts [{'agent_name':'AgentHumility', 'score': 7}, ...]
    """
    if not answers_list:
        return [], 0.0

    transcript = "\n".join(answers_list)
    try:
        scores = await run_analysis_pipeline(transcript)  # expecting list of dicts
    except Exception as e:
        print(f"[run_cumulative_agent_analysis] run_analysis_pipeline failed: {e}")
        scores = []

    # Normalize and compute overall average
    per_agent = []
    total = 0.0
    count = 0
    for item in scores:
        if isinstance(item, dict):
            agent_name = item.get("agent_name", "Unknown")
            score_val = item.get("score", 0)
            # try to parse numeric scores
            try:
                score_num = float(score_val)
            except Exception:
                try:
                    score_num = float(int(score_val))
                except Exception:
                    score_num = 0.0
            per_agent.append({"agent_name": agent_name, "score": score_num})
            total += score_num
            count += 1

    overall_avg = (total / count) if count > 0 else 0.0
    return per_agent, overall_avg

# Use InterviewAnalyzer to keep per-turn detailed analyses for the final report
async def analyze_and_store_turn(question, answer_text):
    global analyzer
    if analyzer is None:
        analyzer = InterviewAnalyzer(candidate_name=getattr(session, "candidate_name", "Candidate"))
    try:
        # analyzer.analyze_response appends the analyzed turn to analyzer.analysis.turns internally
        _ = await analyzer.analyze_response(question, answer_text)
    except Exception as e:
        print(f"[analyze_and_store_turn] error: {e}")

# --- Gradio UI handlers ---

def start_interview(name: str):
    """Initialize session and analyzer."""
    session.reset()
    session.candidate_name = name.strip() or "Candidate"
    session.turns = []  # ensure list exists
    session.start_time = datetime.now().isoformat()
    # Instantiate analyzer
    global analyzer
    analyzer = InterviewAnalyzer(candidate_name=session.candidate_name)
    # initial UI states
    first_q = QUESTIONS[0]
    status_msg = f"Hello {session.candidate_name}. Starting interview."
    return status_msg, first_q, gr.update(visible=True), gr.update(visible=True), ""

async def submit_answer(audio_data):
    """
    Called when user clicks Submit Answer.
    - transcribe
    - append to session.turns
    - run per-turn detailed analysis (stores into analyzer)
    - run cumulative agent analysis across answers so far and return formatted live breakdown
    """
    if audio_data is None:
        return "No audio received. Please record an answer.", "", "", gr.update(visible=True), gr.update(visible=True)

    # transcribe
    transcript = await transcribe(audio_data)

    if transcript.strip() == "":
        return "Could not transcribe audio. Try again.", "", "", gr.update(visible=True), gr.update(visible=True)

    # store turn (simple Turn dataclass usage)
    # idx = len(session.turns)
    # turn = Turn(question=QUESTIONS[idx], audio_data=audio_data, transcript=transcript)
    # session.turns.append(turn)
    current_index = len(session.turns)

    if current_index < len(QUESTIONS):
        turn = Turn(
            question=QUESTIONS[current_index],
            audio_data=audio_data,
            transcript=transcript
        )
        session.turns.append(turn)
    else:
        # All  questions answered — trigger final report generation
        final_results = analyze_answers([t.transcript for t in session.turns])
        pdf_path = generate_pdf_report(final_results)
        return None, final_results, pdf_path



    # Fire and forget per-turn detailed analyzer (but await to ensure stored)
    await analyze_and_store_turn(turn.question, turn.transcript)

    # Build answers list so far for cumulative analysis
    answers_so_far = [t.transcript for t in session.turns]

    per_agent, overall_avg = await run_cumulative_agent_analysis(answers_so_far)

    # Format live progress text (markdown)
    md = f"### Live Progress (after {len(answers_so_far)} answer(s))\n\n"
    md += f"**Cumulative Overall Average:** **{overall_avg:.2f} / 10**\n\n"
    md += "| Agent | Score |\n|---:|:---:|\n"
    for p in per_agent:
        md += f"| {p['agent_name']} | {p['score']:.2f} |\n"

    # prepare next question or finish
    if len(session.turns) < len(QUESTIONS):
        next_q = QUESTIONS[len(session.turns)]
        status = f"Recorded Q{len(session.turns)}. Moving to next question."
        show_generate = False
    else:
        next_q = "All questions completed. Click 'Generate Final Report' to create PDF/HTML."
        status = "Interview complete."
        show_generate = True

    return transcript, next_q, md, gr.update(visible=show_generate), gr.update(visible=show_generate)

async def generate_final_report_and_pdf():
    """
    Create final HTML via analyzer.generate_report and convert to PDF.
    Returns (html_str, html_path, pdf_path_or_none)
    """
    if analyzer is None or not analyzer.analysis.turns:
        return "No interview data available. Please run the interview first.", None, None

    # Make sure overall scores are computed
    html = analyzer.generate_report(format="html")
    html_path = report_generator.save_html_report(html, analyzer.candidate_name)
    pdf_path = report_generator.create_pdf_report_from_html(html, analyzer.candidate_name)

    return html, html_path, pdf_path

# --- Gradio UI ---

with gr.Blocks(title="🎙️ Interview + Live Cumulative Analysis + PDF Report") as demo:
    gr.Markdown("## Interview Assistant — live per-agent breakdown and final PDF report")

    with gr.Row():
        with gr.Column(scale=1):
            name_input = gr.Textbox(label="Candidate Name", placeholder="Enter candidate name")
            start_btn = gr.Button("Start Interview")

            question_display = gr.Textbox(label="Current Question", interactive=False, lines=3)
            audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record your answer", visible=False)
            submit_btn = gr.Button("Submit Answer", visible=False)
            generate_report_btn = gr.Button("Generate Final Report (HTML + PDF)", visible=False)

            status = gr.Markdown("")

        with gr.Column(scale=1):
            transcript_box = gr.Textbox(label="Last Transcript", lines=4, interactive=False)
            live_progress = gr.Markdown("### Live progress will appear here")
            final_report_html = gr.HTML("")
            download_html = gr.File(label="Download HTML report")
            download_pdf = gr.File(label="Download PDF report")

    # Start interview
    start_btn.click(
        fn=start_interview,
        inputs=[name_input],
        outputs=[status, question_display, audio_input, submit_btn, live_progress]
    )

    # When user clicks Submit Answer
    submit_btn.click(
        fn=submit_answer,
        inputs=[audio_input],
        outputs=[transcript_box, question_display, live_progress, generate_report_btn, generate_report_btn]
    )

    # Generate final report
    generate_report_btn.click(
        fn=generate_final_report_and_pdf,
        inputs=None,
        outputs=[final_report_html, download_html, download_pdf]
    )

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860)
