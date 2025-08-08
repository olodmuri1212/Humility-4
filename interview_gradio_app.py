# #!/usr/bin/env python3
# # interview_gradio_app.py

# import os
# import asyncio
# import numpy as np
# import torch
# import gradio as gr
# import librosa
# from faster_whisper import WhisperModel

# from interview_state import SessionState, Turn
# from backend.agent_manager import run_analysis_pipeline, AnalysisResult
# from services.report_generator import generate_html_report, create_pdf_report

# # Question bank
# QUESTIONS = [
#     "1) Can you tell me about a time when you received constructive criticism? How did you handle it?",
#     "2) Describe a situation where you had to work with a difficult team member. How did you handle it?",
#     "3) Tell me about a time when you made a mistake at work. How did you address it?",
#     "4) How do you handle situations where you need to learn something new?",
#     "5) Can you share an example of when you had to adapt to a significant change at work?"
# ]

# # Whisper model setup
# MODEL_SIZE = "base"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# whisper_model = WhisperModel(
#     MODEL_SIZE,
#     device=DEVICE,
#     compute_type="int8" if DEVICE == "cuda" else "int8"
# )

# # Global session state
# session = SessionState()


# def preprocess_audio(audio_data):
#     """Convert Gradio audio output to float32 16kHz mono."""
#     if audio_data is None:
#         return np.zeros((0,), dtype=np.float32)
    
#     sr, arr = audio_data
#     if arr.ndim > 1 and arr.shape[1] == 2:
#         arr = arr.mean(axis=1)
#     if arr.dtype != np.float32:
#         arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
#     if sr != 16000:
#         arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
#     return arr


# async def transcribe_turn(turn: Turn):
#     """Run Whisper on a single Turn."""
#     audio = preprocess_audio(turn.audio_data)
#     segments, _ = whisper_model.transcribe(
#         audio,
#         language="en",
#         beam_size=5,
#         vad_filter=True
#     )
#     turn.transcript = " ".join([seg.text for seg in segments]).strip()


# async def analyze_turn(turn: Turn):
#     """Run analysis pipeline on a single Turn."""
#     results = await run_analysis_pipeline(turn.transcript)
#     turn.analysis_results = results


# def start_interview(name: str):
#     """Initialize a new interview session."""
#     session.reset()
#     session.candidate_name = name.strip() or "Candidate"
    
#     return (
#         f"Hello {session.candidate_name}, let's begin!",
#         QUESTIONS[0],
#         gr.update(visible=True),
#         gr.update(visible=True),
#         ""
#     )


# def record_answer(audio_data):
#     """Process a recorded answer and prepare for next question."""
#     idx = len(session.turns)
#     session.turns.append(Turn(question=QUESTIONS[idx], audio_data=audio_data))
    
#     # Transcribe immediately
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     loop.run_until_complete(transcribe_turn(session.turns[idx]))
    
#     # Build running transcript list
#     transcripts = "\n\n".join(
#         f"Q: {t.question}\nA: {t.transcript if t.transcript else '[Not transcribed yet]'}"
#         for t in session.turns
#     )
    
#     if idx + 1 < len(QUESTIONS):
#         return (
#             f"Recorded answer {idx+1}/{len(QUESTIONS)}. Next question:",
#             QUESTIONS[idx+1],
#             gr.update(visible=True),
#             gr.update(visible=True),
#             transcripts
#         )
#     else:
#         return (
#             "All questions recorded!",
#             "",
#             gr.update(visible=False),
#             gr.update(visible=False),
#             transcripts
#         )


# def generate_report_ui():
#     """Generate and display the interview report."""
#     if not session.turns or not session.turns[0].transcript:
#         return "No transcripts available. Please record and transcribe answers first.", None, None
    
#     # Run analysis on all turns
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     tasks = [analyze_turn(t) for t in session.turns]
#     loop.run_until_complete(asyncio.gather(*tasks))
    
#     # Compute final scores
#     session.compute_session_scores()
    
#     # Build report data structure
#     from core.state import InterviewState, ConversationTurn, AgentScore
    
#     core_state = InterviewState(
#         candidate_name=session.candidate_name,
#         normalized_humility_score=session.normalized_humility_score,
#         cumulative_scores=session.cumulative_scores,
#         conversation_history=[]
#     )
    
#     for turn in session.turns:
#         conv_turn = ConversationTurn(
#             question=turn.question,
#             transcript=turn.transcript,
#             analysis_results=[
#                 AgentScore(
#                     agent_name=r.agent_name,
#                     score=r.score,
#                     evidence=r.evidence
#                 )
#                 for r in turn.analysis_results
#             ]
#         )
#         core_state.conversation_history.append(conv_turn)
    
#     # Generate reports
#     html = generate_html_report(core_state)
#     pdf_bytes = create_pdf_report(core_state)
    
#     # Save reports
#     os.makedirs("reports", exist_ok=True)
#     base_path = os.path.join("reports", session.candidate_name.replace(" ", "_"))
#     html_path = f"{base_path}_report.html"
#     pdf_path = f"{base_path}_report.pdf"
    
#     with open(html_path, "w", encoding="utf-8") as f:
#         f.write(html)
#     with open(pdf_path, "wb") as f:
#         f.write(pdf_bytes)
    
#     return html, html_path, pdf_path


# # Gradio UI
# with gr.Blocks(title="🎙️ Humility Interview Assistant") as demo:
#     gr.Markdown("## AI-Powered Humility Interview Practice")
    
#     with gr.Row():
#         with gr.Column():
#             # Input section
#             name_in = gr.Textbox(label="Your Name", placeholder="Enter your name")
#             start_btn = gr.Button("▶ Start Interview")
            
#             # Question display
#             question_box = gr.Textbox(
#                 label="Current Question",
#                 interactive=False,
#                 lines=3
#             )
            
#             # Audio recording
#             audio_in = gr.Audio(
#                 label="Record your answer",
#                 type="numpy",
#                 visible=False
#             )
            
#             # Action buttons
#             submit_btn = gr.Button("Submit Answer", visible=False)
#             status = gr.Markdown("")
            
#             # Transcripts
#             transcripts_out = gr.Textbox(
#                 label="Interview Transcripts",
#                 lines=10,
#                 interactive=False
#             )
            
#             # Report generation
#             generate_report_btn = gr.Button("▶ Generate Report", visible=False)
#             report_html = gr.HTML()
#             download_html = gr.File(label="Download HTML Report", visible=False)
#             download_pdf = gr.File(label="Download PDF Report", visible=False)
    
#     # Event handlers
#     start_btn.click(
#         fn=start_interview,
#         inputs=[name_in],
#         outputs=[
#             status, 
#             question_box, 
#             audio_in, 
#             submit_btn, 
#             transcripts_out
#         ]
#     )
    
#     submit_btn.click(
#         fn=record_answer,
#         inputs=[audio_in],
#         outputs=[
#             status,
#             question_box,
#             audio_in,
#             submit_btn,
#             transcripts_out
#         ]
#     ).then(
#         fn=lambda: gr.update(visible=len(session.turns) == len(QUESTIONS)),
#         outputs=[generate_report_btn]
#     )
    
#     generate_report_btn.click(
#         fn=generate_report_ui,
#         inputs=None,
#         outputs=[report_html, download_html, download_pdf]
#     ).then(
#         fn=lambda: gr.update(visible=True),
#         outputs=[download_html, download_pdf]
#     )


# if __name__ == "__main__":
#     demo.launch(server_name="localhost", server_port=7860)
































import os
import numpy as np
import torch
import gradio as gr
import librosa
import asyncio
from faster_whisper import WhisperModel

from interview_analyzer import InterviewAnalyzer

# 1) Configuration
QUESTIONS = [
    "1) Can you tell me about a time when you received constructive criticism? How did you handle it?",
    "2) Describe a situation where you had to work with a difficult team member. How did you handle it?",
    "3) Tell me about a time when you made a mistake at work. How did you address it?",
    "4) How do you handle situations where you need to learn something new?",
    "5) Can you share an example of when you had to adapt to a significant change at work?"
]

# 2) Whisper STT setup
MODEL_SIZE = "base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type="int8" if DEVICE == "cuda" else "int8"
)


def preprocess_audio(audio_data):
    """Convert Gradio audio numpy → float32 mono @16kHz."""
    if audio_data is None:
        return np.zeros((0,), dtype=np.float32)
    sr, arr = audio_data
    # stereo→mono
    if arr.ndim > 1 and arr.shape[1] == 2:
        arr = arr.mean(axis=1)
    # ints→float32
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
    # resample to 16kHz
    if sr != 16000:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    return arr


async def transcribe(audio_data):
    """Run Whisper on a single recording."""
    audio = preprocess_audio(audio_data)
    segments, _ = whisper_model.transcribe(
        audio, language="en", beam_size=5, vad_filter=True
    )
    return " ".join(seg.text for seg in segments).strip()


# 3) Global interview state
class UIState:
    def __init__(self):
        self.analyzer: InterviewAnalyzer = None
        self.current_q = 0
        self.transcripts: list[str] = []


state = UIState()


# 4) Gradio callbacks
def start_interview(name: str):
    """Initialize InterviewAnalyzer and reset state."""
    analyzer = InterviewAnalyzer(candidate_name=name.strip() or "Candidate")
    state.analyzer = analyzer
    state.current_q = 0
    state.transcripts = []
    return (
        f"Hello {analyzer.candidate_name}, let's begin!",
        QUESTIONS[0],
        gr.update(visible=True),  # show recorder
        gr.update(visible=True),  # show submit
        "",                       # transcript log
        gr.update(visible=False)  # hide report button
    )


async def submit_answer(audio_data):
    """Transcribe this answer, run analysis, advance to next."""
    if state.analyzer is None:
        return (
            "Please click Start first.",
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            "",
            gr.update(visible=False)
        )

    # 1) Transcribe
    text = await transcribe(audio_data)
    state.transcripts.append(text)

    # 2) Analyze with InterviewAnalyzer
    analysis = await state.analyzer.analyze_response(
        QUESTIONS[state.current_q], text
    )

    # 3) Build running transcript+score summary
    log_lines = []
    for i, turn in enumerate(state.analyzer.analysis.turns):
        q = turn.question
        t = turn.answer
        s = turn.analysis["overall_score"]
        log_lines.append(f"Q{i+1}: {t}\n→ Score: {s:.1f}/10\n")
    transcript_log = "\n".join(log_lines)

    # 4) Advance or finish
    state.current_q += 1
    if state.current_q < len(QUESTIONS):
        return (
            f"Recorded answer {state.current_q}.",
            QUESTIONS[state.current_q],
            gr.update(visible=True),
            gr.update(visible=True),
            transcript_log,
            gr.update(visible=False)
        )
    else:
        return (
            "All questions answered! Click ▶ Generate Report",
            "",
            gr.update(visible=False),
            gr.update(visible=False),
            transcript_log,
            gr.update(visible=True)
        )


def generate_report():
    """Ask InterviewAnalyzer for the final report."""
    if state.analyzer is None or not state.analyzer.analysis.turns:
        return ("No data to generate report.", None, None)

    # 1) Generate HTML report
    html = state.analyzer.generate_report(format="html")

    # 2) Save HTML
    os.makedirs("reports", exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in state.analyzer.candidate_name)
    path_html = f"reports/report_{safe}.html"
    with open(path_html, "w", encoding="utf-8") as f:
        f.write(html)

    # 3) Also save plain text if you like
    txt = state.analyzer.generate_report(format="text")
    path_txt = path_html.replace(".html", ".txt")
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(txt)

    return html, path_html, path_txt


# 5) Build Gradio UI
with gr.Blocks(title="🎙️ Humility Interview Assistant") as demo:
    gr.Markdown("## AI-Powered Humility Interview Practice")
    with gr.Column():
        name_in = gr.Textbox(label="Enter your name")
        start_btn = gr.Button("▶ Start Interview")
        question_box = gr.Textbox(label="Current Question", interactive=False, lines=2)
        audio_in = gr.Audio(label="Record your answer", type="numpy", visible=False)
        submit_btn = gr.Button("Submit Answer", visible=False)
        transcript_log = gr.Textbox(label="Transcript & Scores so far", lines=8, interactive=False)
        report_btn = gr.Button("▶ Generate Report", visible=False)
        report_view = gr.HTML()
        download_html = gr.File(label="Download HTML Report")
        download_txt = gr.File(label="Download Text Report")

    start_btn.click(
        start_interview,
        inputs=[name_in],
        outputs=[report_view, question_box, audio_in, submit_btn, transcript_log, report_btn]
    )
    submit_btn.click(
        submit_answer,
        inputs=[audio_in],
        outputs=[report_view, question_box, audio_in, submit_btn, transcript_log, report_btn]
    )
    report_btn.click(
        generate_report,
        inputs=None,
        outputs=[report_view, download_html, download_txt]
    )

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860)
