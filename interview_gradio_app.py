

# import os
# import numpy as np
# import torch
# import gradio as gr
# import librosa
# import asyncio
# from faster_whisper import WhisperModel
# from interview_state import SessionState, Turn
# from interview_analyzer import InterviewAnalyzer
# from services.report_generator import generate_html_report
# from backend.agent_manager import run_analysis_pipeline
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


# class UIState:
#     def __init__(self):
#         self.analyzer: InterviewAnalyzer | None = None
#         self.current_q: int = 0


# state_ui = UIState()


# # Helpers
# def preprocess_audio(audio_data):
#     """Convert gr.Audio numpy → float32 mono @16kHz."""
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


# async def transcribe(audio_data):
#     """Run Whisper on raw audio."""
#     audio = preprocess_audio(audio_data)
#     segments, _ = whisper_model.transcribe(
#         audio,
#         language="en",
#         beam_size=5,
#         vad_filter=True
#     )
#     return " ".join(seg.text for seg in segments).strip()


# async def analyze_turn(turn: Turn):
#     """Run analysis pipeline on a single Turn."""
#     results = await run_analysis_pipeline(turn.transcript)
#     turn.analysis_results = results


# def start_interview(name: str):
#     """Initialize a new interview session."""
#     session.reset()
#     session.candidate_name = name.strip() or "Candidate"
#     state_ui.analyzer = InterviewAnalyzer(candidate_name=session.candidate_name)  # Ensure this is initialized
#     state_ui.current_q = 0

#     return (
#         f"Hello {session.candidate_name}, let's begin!",
#         QUESTIONS[0],
#         gr.update(visible=True),
#         gr.update(visible=True),
#         ""
#     )


# async def record_answer(audio_data):
#     """Process a recorded answer and prepare for the next question."""
#     idx = len(session.turns)

#     # Check if the index is within the range of QUESTIONS
#     if idx >= len(QUESTIONS):
#         return (
#             "All questions recorded! Click 'Generate Report' to view your results.",
#             "",
#             ""
#         )

#     session.turns.append(
#         Turn(
#             question=QUESTIONS[idx],
#             audio_data=audio_data
#         )
#     )

#     # Transcribe immediately
#     transcript = await transcribe(audio_data)
#     session.turns[-1].transcript = transcript

#     # Analyze the response
#     await analyze_turn(session.turns[idx])  # Ensure this is called

#     # Build running transcript list
#     transcripts = "\n\n".join(
#         f"Q: {t.question}\nA: {t.transcript if t.transcript else '[Not transcribed yet]'}"
#         for t in session.turns
#     )

#     if idx + 1 < len(QUESTIONS):
#         return (
#             f"Recorded answer {idx + 1}/{len(QUESTIONS)}. Next question:",
#             QUESTIONS[idx + 1],
#             transcripts
#         )
#     else:
#         return (
#             "All questions recorded! Click 'Generate Report' to view your results.",
#             "",
#             transcripts
#         )


# def generate_report():
#     """Generate a report from all interview responses."""
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

#     # Save reports
#     os.makedirs("reports", exist_ok=True)
#     base_path = os.path.join("reports", session.candidate_name.replace(" ", "_"))
#     html_path = f"{base_path}_report.html"

#     with open(html_path, "w", encoding="utf-8") as f:
#         f.write(html)

#     return html, html_path


# Ensure the report button is connected correctly



# Gradio UI
# with gr.Blocks(title="🎙️ Humility Interview Assistant") as demo:
#     gr.Markdown("## AI-Powered Humility Interview Practice")

#     with gr.Row():
#         with gr.Column(scale=1):
#             name_input = gr.Textbox(label="Candidate Name", placeholder="Enter candidate name")
#             question_display = gr.Markdown("## " + QUESTIONS[0])
#             audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record your answer")
#             submit_btn = gr.Button("Submit Answer")
#             next_btn = gr.Button("Next Question", visible=True)
#             report_btn = gr.Button("Generate Report", visible=False)
#             status = gr.Markdown("")

#         with gr.Column(scale=1):
#             transcript_display = gr.Textbox(label="Transcript", lines=10, interactive=False)
#             analysis_display = gr.Markdown("## Analysis will appear here")
#             report_display = gr.Markdown("", visible=False)




# # Gradio UI
# with gr.Blocks(title="🎙️ Humility Interview Assistant") as demo:
#     gr.Markdown("## AI-Powered Humility Interview Practice")

#     with gr.Row():
#         with gr.Column(scale=1):
#             name_input = gr.Textbox(label="Candidate Name", placeholder="Enter candidate name")
#             start_btn = gr.Button("▶ Start Interview")

#             question_display = gr.Markdown("## " + QUESTIONS[0])
#             audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record your answer")
#             submit_btn = gr.Button("Submit Answer")
#             next_btn = gr.Button("Next Question", visible=False)
#             report_btn = gr.Button("Generate Report", visible=False)
#             status = gr.Markdown("")

#         with gr.Column(scale=1):
#             transcript_display = gr.Textbox(label="Transcript", lines=10, interactive=False)
#             analysis_display = gr.Markdown("## Analysis will appear here")
#             report_display = gr.Markdown("", visible=False)
#             download_html = gr.File(label="Download HTML Report", visible=False)  # Define download_html
#             download_pdf = gr.File(label="Download PDF Report", visible=False)    # Define download_pdf

#     # Event handlers
#     start_btn.click(
#         fn=start_interview,
#         inputs=[name_input],
#         outputs=[transcript_display, question_display, audio_input, submit_btn, next_btn]
#     )

#     submit_btn.click(
#         fn=record_answer,
#         inputs=[audio_input],
#         outputs=[transcript_display, question_display, audio_input, submit_btn, next_btn]
#     )

#     next_btn.click(
#         fn=on_next,
#         inputs=[],
#         outputs=[question_display, next_btn, report_btn]
#     )

#     report_btn.click(
#         fn=generate_report,
#         inputs=None,
#         outputs=[report_display, download_html, download_pdf]
#     )


#     # Event handlers
#     # def update_name(name):
#     #     state_ui.analyzer.candidate_name = name
#     #     return name
#     def update_name(name):           
#         if state_ui.analyzer is None:
#         # Initialize the analyzer
              
#             state_ui.analyzer =  InterviewAnalyzer(candidate_name=name)  # Ensure this is initialized                      
                                 
#         state_ui.analyzer.candidate_name = name                       
#         return name   

#     async def on_submit(audio_path):
#         if not audio_path:
#             return "", "Please record an answer before submitting.", ""
#         return await record_answer(audio_path)

#     # def on_next():
#     #     if state_ui.current_q < len(QUESTIONS):
#     #         return {
#     #             question_display: gr.Markdown.update(value=f"## {QUESTIONS[state_ui.current_q]}"),
#     #             next_btn: gr.Button.update(visible=state_ui.current_q < len(QUESTIONS) - 1),
#     #             report_btn: gr.Button.update(visible=state_ui.current_q >= len(QUESTIONS) - 1)
#     #         }
#     def on_next():
#         """Move to the next question."""
#         state_ui.current_q += 1
#         is_last_question = state_ui.current_q >= len(QUESTIONS) - 1

#         return {
#             question_display: gr.Markdown.update(
#                 value=f"## {QUESTIONS[state_ui.current_q]}"
#             ),  # Update the question display correctly
#             next_btn: gr.Button.update(
#                 visible=state_ui.current_q < len(QUESTIONS) - 1
#             ),
#             report_btn: gr.Button.update(
#             visible=state_ui.current_q >= len(QUESTIONS) - 1
#         )
#     }                   

#     def show_report():
#         return {
#             report_display: gr.Markdown.update(
#                 value=generate_report(),
#                 visible=True
#             )
#         }

#     # Connect UI components to handlers
#     name_input.change(fn=update_name, inputs=name_input, outputs=name_input)
#     submit_btn.click(fn=on_submit, inputs=[audio_input], outputs=[transcript_display, status, next_btn])
#     next_btn.click(fn=on_next, inputs=[], outputs=[question_display, next_btn, report_btn])
#     #report_btn.click(fn=show_report, inputs=[], outputs=[report_display])
#     report_btn.click(
#     fn=generate_report,
#     inputs=None,
#     outputs=[report_display, download_html, download_pdf]
#     )

# if __name__ == "__main__":
#     demo.launch(server_name="localhost", server_port=7860)
























# ##working code 3
# import os
# import numpy as np
# import torch
# import gradio as gr
# import librosa
# import asyncio
# from faster_whisper import WhisperModel
# from interview_state import SessionState, Turn
# from interview_analyzer import InterviewAnalyzer
# from backend.agent_manager import run_analysis_pipeline

# # ===============================
# # Question bank
# # ===============================
# QUESTIONS = [
#     "1) Can you tell me about a time when you received constructive criticism? How did you handle it?",
#     "2) Describe a situation where you had to work with a difficult team member. How did you handle it?",
#     "3) Tell me about a time when you made a mistake at work. How did you address it?",
#     "4) How do you handle situations where you need to learn something new?",
#     "5) Can you share an example of when you had to adapt to a significant change at work?"
# ]

# # ===============================
# # Whisper model setup
# # ===============================
# MODEL_SIZE = "base"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# whisper_model = WhisperModel(
#     MODEL_SIZE,
#     device=DEVICE,
#     compute_type="int8" if DEVICE == "cuda" else "int8"
# )

# # ===============================
# # Global session & UI state
# # ===============================
# session = SessionState()

# class UIState:
#     def __init__(self):
#         self.analyzer: InterviewAnalyzer | None = None
#         self.current_q: int = 0

# state_ui = UIState()

# # ===============================
# # Helper Functions
# # ===============================
# def preprocess_audio(audio_data):
#     """Convert gr.Audio numpy → float32 mono @16kHz."""
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

# async def transcribe(audio_data):
#     """Run Whisper on raw audio."""
#     audio = preprocess_audio(audio_data)
#     segments, _ = whisper_model.transcribe(
#         audio,
#         language="en",
#         beam_size=5,
#         vad_filter=True
#     )
#     return " ".join(seg.text for seg in segments).strip()

# def start_interview(name: str):
#     """Initialize a new interview session."""
#     session.reset()
#     session.candidate_name = name.strip() or "Candidate"
#     state_ui.analyzer = InterviewAnalyzer(candidate_name=session.candidate_name)
#     state_ui.current_q = 0

#     return (
#         f"Hello {session.candidate_name}, let's begin!",
#         QUESTIONS[0],
#         gr.update(visible=True),
#         gr.update(visible=True),
#         "",
#         gr.update(visible=False),
#         gr.update(visible=False)
#     )

# async def record_answer(audio_data):
#     """Process a recorded answer and prepare for the next question."""
#     idx = len(session.turns)

#     if idx >= len(QUESTIONS):
#         return (
#             "All questions recorded! Click 'Show Transcriptions' or 'Generate Report'.",
#             "",
#             "",
#             gr.update(visible=False),
#             gr.update(visible=False),
#             gr.update(visible=True),
#             gr.update(visible=True)
#         )

#     session.turns.append(
#         Turn(
#             question=QUESTIONS[idx],
#             audio_data=audio_data
#         )
#     )

#     # Transcribe immediately
#     transcript = await transcribe(audio_data)
#     session.turns[-1].transcript = transcript

#     # Build running transcript list
#     transcripts = "\n\n".join(
#         f"Q: {t.question}\nA: {t.transcript if t.transcript else '[Not transcribed yet]'}"
#         for t in session.turns
#     )

#     if idx + 1 < len(QUESTIONS):
#         return (
#             f"Recorded answer {idx + 1}/{len(QUESTIONS)}. Next question:",
#             QUESTIONS[idx + 1],
#             transcripts,
#             gr.update(visible=True),
#             gr.update(visible=True),
#             gr.update(visible=True),
#             gr.update(visible=True)
#         )
#     else:
#         return (
#             "All questions recorded! Click 'Show Transcriptions' or 'Generate Report'.",
#             "",
#             transcripts,
#             gr.update(visible=False),
#             gr.update(visible=False),
#             gr.update(visible=True),
#             gr.update(visible=True)
#         )

# def show_transcriptions():
#     """Display transcriptions for all interview responses."""
#     if not session.turns or not session.turns[0].transcript:
#         return "No transcripts available. Please record and transcribe answers first."

#     transcriptions_display = "\n\n".join(
#         f"Q: {turn.question}\nA: {turn.transcript}"
#         for turn in session.turns
#     )
#     return transcriptions_display

# # ===============================
# # Analysis logic (merged from mangoo.py)
# # ===============================
# async def analyze_answers(candidate_name, *answers):
#     transcript = "\n".join(answers)
#     scores = await run_analysis_pipeline(transcript)

#     detailed_scores = []
#     overall_score = 0

#     for score in scores:
#         if isinstance(score, dict):
#             agent_name = score.get('agent_name', 'Unknown Agent')
#             score_value = score.get('score', 0)
#             if isinstance(score_value, str):
#                 try:
#                     score_value = int(score_value)
#                 except ValueError:
#                     score_value = 0
#             detailed_scores.append((agent_name, score_value))
#             overall_score += score_value
#     overall_score = overall_score / len(scores) if scores else 0
#     return detailed_scores, overall_score

# def format_results(detailed_scores, overall_score):
#     result_text = ""
#     for agent_name, score in detailed_scores:
#         if score > 7:
#             result_text += f"{agent_name}: High Score ({score})\n"
#         elif score > 3:
#             result_text += f"{agent_name}: Medium Score ({score})\n"
#         else:
#             result_text += f"{agent_name}: Low Score ({score})\n"
#     result_text += f"\nOverall Combined Score: {overall_score:.2f}\n"
#     return result_text

# async def generate_final_report():
#     """Generate final report from recorded transcripts."""
#     if not session.turns:
#         return "No data available for analysis."

#     answers = [turn.transcript for turn in session.turns]
#     detailed_scores, overall_score = await analyze_answers(session.candidate_name, *answers)
#     return format_results(detailed_scores, overall_score)

# # ===============================
# # Gradio UI
# # ===============================
# with gr.Blocks(title="🎙️ Interview & Report Generator") as demo:
#     gr.Markdown("## AI-Powered Interview + Final Report")

#     with gr.Row():
#         with gr.Column(scale=1):
#             name_input = gr.Textbox(label="Candidate Name", placeholder="Enter candidate name")
#             start_btn = gr.Button("▶ Start Interview")
#             question_display = gr.Markdown("## " + QUESTIONS[0])
#             audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record your answer")
#             submit_btn = gr.Button("Submit Answer")
#             transcription_btn = gr.Button("Show Transcriptions", visible=False)
#             report_btn = gr.Button("Generate Final Report", visible=False)
#             status = gr.Markdown("")

#         with gr.Column(scale=1):
#             transcript_display = gr.Textbox(label="Transcript", lines=10, interactive=False)
#             transcriptions_display = gr.Markdown("", visible=False)
#             report_display = gr.Textbox(label="Final Report", interactive=False)

#     # Event handlers
#     start_btn.click(
#         fn=start_interview,
#         inputs=[name_input],
#         outputs=[transcript_display, question_display, audio_input, submit_btn, transcription_btn, transcription_btn, report_btn]
#     )

#     submit_btn.click(
#         fn=record_answer,
#         inputs=[audio_input],
#         outputs=[transcript_display, question_display, transcript_display, audio_input, submit_btn, transcription_btn, report_btn]
#     )

#     transcription_btn.click(
#         fn=show_transcriptions,
#         outputs=[transcriptions_display]
#     )

#     report_btn.click(
#         fn=generate_final_report,
#         outputs=[report_display]
#     )

# if __name__ == "__main__":
#     demo.launch(server_name="localhost", server_port=7860, share=True)





















##chatgpt-5 first trial
# import os
# import numpy as np
# import torch
# import gradio as gr
# import librosa
# from faster_whisper import WhisperModel
# from interview_state import SessionState, Turn
# from interview_analyzer import InterviewAnalyzer
# from backend.agent_manager import run_analysis_pipeline

# # ===============================
# # Question bank
# # ===============================
# QUESTIONS = [
#     "1) Can you tell me about a time when you received constructive criticism? How did you handle it?",
#     "2) Describe a situation where you had to work with a difficult team member. How did you handle it?",
#     "3) Tell me about a time when you made a mistake at work. How did you address it?",
#     "4) How do you handle situations where you need to learn something new?",
#     "5) Can you share an example of when you had to adapt to a significant change at work?"
# ]

# # ===============================
# # Whisper model setup
# # ===============================
# MODEL_SIZE = "base"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# whisper_model = WhisperModel(
#     MODEL_SIZE,
#     device=DEVICE,
#     compute_type="int8" if DEVICE == "cuda" else "int8"
# )

# # ===============================
# # Global session & UI state
# # ===============================
# session = SessionState()

# class UIState:
#     def __init__(self):
#         self.analyzer: InterviewAnalyzer | None = None
#         self.current_q: int = 0

# state_ui = UIState()

# # ===============================
# # Helper Functions
# # ===============================
# def preprocess_audio(audio_data):
#     """Convert gr.Audio numpy → float32 mono @16kHz."""
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

# def transcribe(audio_data):
#     """Run Whisper on raw audio."""
#     audio = preprocess_audio(audio_data)
#     segments, _ = whisper_model.transcribe(
#         audio,
#         language="en",
#         beam_size=5,
#         vad_filter=True
#     )
#     return " ".join(seg.text for seg in segments).strip()

# def start_interview(name: str):
#     """Initialize a new interview session."""
#     session.reset()
#     session.candidate_name = name.strip() or "Candidate"
#     state_ui.analyzer = InterviewAnalyzer(candidate_name=session.candidate_name)
#     state_ui.current_q = 0

#     return (
#         f"Hello {session.candidate_name}, let's begin!",
#         QUESTIONS[0],
#         gr.update(visible=True),
#         gr.update(visible=True),
#         "",
#         gr.update(visible=False),
#         ""
#     )

# def record_answer(audio_data):
#     """Process a recorded answer and update transcript panel."""
#     idx = len(session.turns)

#     if idx >= len(QUESTIONS):
#         return (
#             "All questions recorded! Click 'Generate Final Report'.",
#             "",
#             "",
#             gr.update(visible=False),
#             gr.update(visible=False),
#             gr.update(visible=True),
#             get_all_transcripts()
#         )

#     session.turns.append(
#         Turn(
#             question=QUESTIONS[idx],
#             audio_data=audio_data
#         )
#     )

#     transcript = transcribe(audio_data)
#     session.turns[-1].transcript = transcript

#     # Build running transcript list for live display
#     transcripts_text = get_all_transcripts()

#     if idx + 1 < len(QUESTIONS):
#         return (
#             f"Recorded answer {idx + 1}/{len(QUESTIONS)}. Next question:",
#             QUESTIONS[idx + 1],
#             transcripts_text,
#             gr.update(visible=True),
#             gr.update(visible=True),
#             gr.update(visible=True),
#             transcripts_text
#         )
#     else:
#         return (
#             "All questions recorded! Click 'Generate Final Report'.",
#             "",
#             transcripts_text,
#             gr.update(visible=False),
#             gr.update(visible=False),
#             gr.update(visible=True),
#             transcripts_text
#         )

# def get_all_transcripts():
#     """Return full Q/A transcript text."""
#     return "\n\n".join(
#         f"Q: {t.question}\nA: {t.transcript if t.transcript else '[Not transcribed yet]'}"
#         for t in session.turns
#     )

# # ===============================
# # Analysis logic (from mangoo.py)
# # ===============================
# def analyze_answers(candidate_name, *answers):
#     transcript = "\n".join(answers)
#     scores = run_analysis_pipeline(transcript)

#     detailed_scores = []
#     overall_score = 0

#     for score in scores:
#         if isinstance(score, dict):
#             agent_name = score.get('agent_name', 'Unknown Agent')
#             score_value = score.get('score', 0)
#             if isinstance(score_value, str):
#                 try:
#                     score_value = int(score_value)
#                 except ValueError:
#                     score_value = 0
#             detailed_scores.append((agent_name, score_value))
#             overall_score += score_value
#     overall_score = overall_score / len(scores) if scores else 0
#     return detailed_scores, overall_score

# def format_results(detailed_scores, overall_score):
#     result_text = ""
#     for agent_name, score in detailed_scores:
#         if score > 7:
#             result_text += f"{agent_name}: High Score ({score})\n"
#         elif score > 3:
#             result_text += f"{agent_name}: Medium Score ({score})\n"
#         else:
#             result_text += f"{agent_name}: Low Score ({score})\n"
#     result_text += f"\nOverall Combined Score: {overall_score:.2f}\n"
#     return result_text

# def generate_final_report():
#     """Generate final report from recorded transcripts."""
#     if not session.turns:
#         return "No data available for analysis."

#     answers = [turn.transcript for turn in session.turns]
#     detailed_scores, overall_score = analyze_answers(session.candidate_name, *answers)
#     return format_results(detailed_scores, overall_score)

# # ===============================
# # Gradio UI
# # ===============================
# with gr.Blocks(title="🎙️ Interview & Report Generator") as demo:
#     gr.Markdown("## AI-Powered Interview + Final Report")

#     with gr.Row():
#         with gr.Column(scale=1):
#             name_input = gr.Textbox(label="Candidate Name", placeholder="Enter candidate name")
#             start_btn = gr.Button("▶ Start Interview")
#             question_display = gr.Markdown("## " + QUESTIONS[0])
#             audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record your answer")
#             submit_btn = gr.Button("Submit Answer")
#             report_btn = gr.Button("Generate Final Report", visible=False)
#             status = gr.Markdown("")

#         with gr.Column(scale=1):
#             transcript_display = gr.Textbox(label="Live Transcripts", lines=20, interactive=False)
#             report_display = gr.Textbox(label="Final Report", interactive=False)

#     # Event handlers
#     start_btn.click(
#         fn=start_interview,
#         inputs=[name_input],
#         outputs=[status, question_display, audio_input, submit_btn, report_btn, transcript_display, transcript_display]
#     )

#     submit_btn.click(
#         fn=record_answer,
#         inputs=[audio_input],
#         outputs=[status, question_display, transcript_display, audio_input, submit_btn, report_btn, transcript_display]
#     )

#     report_btn.click(
#         fn=generate_final_report,
#         outputs=[report_display]
#     )

# if __name__ == "__main__":
#     demo.launch(server_name="localhost", server_port=7860)





















##chatgpt-5o secomnfd trial

# # interview_app.py
# import os
# import numpy as np
# import torch
# import gradio as gr
# import librosa
# import asyncio
# from faster_whisper import WhisperModel
# from datetime import datetime
# from io import BytesIO
# import soundfile as sf

# # Your existing modules (make sure these import paths match your repo)
# from interview_state import SessionState, Turn
# from interview_analyzer import InterviewAnalyzer
# from backend.agent_manager import run_analysis_pipeline  # returns list of dicts [{'agent_name':..., 'score':...}, ...]
 

# # Make sure reports dir exists
# os.makedirs("reports", exist_ok=True)

# # Questions (same as before)
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
# whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8" if DEVICE == "cuda" else "int8")

# # Session/State
# session = SessionState()  # re-use your existing SessionState dataclass
# analyzer: InterviewAnalyzer | None = None

# # --- Helpers ---

# def preprocess_audio(audio_data):
#     """Convert gr.Audio numpy → float32 mono @16kHz."""
#     if audio_data is None:
#         return np.zeros((0,), dtype=np.float32)

#     sr, arr = audio_data
#     if arr is None:
#         return np.zeros((0,), dtype=np.float32)

#     # if stereo, convert to mono
#     if arr.ndim > 1 and arr.shape[1] == 2:
#         arr = arr.mean(axis=1)

#     # normalize to float32
#     if arr.dtype != np.float32:
#         # if integer type, convert to float32
#         if np.issubdtype(arr.dtype, np.integer):
#             arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
#         else:
#             arr = arr.astype(np.float32)

#     if sr != 16000:
#         arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)

#     return arr

# async def transcribe(audio_data):
#     """Run Whisper on raw audio asynchronously."""
#     audio = preprocess_audio(audio_data)
#     try:
#         segments, _ = whisper_model.transcribe(
#             audio,
#             language="en",
#             beam_size=5,
#             vad_filter=True
#         )
#         return " ".join(seg.text for seg in segments).strip()
#     except Exception as e:
#         print(f"[transcribe] error: {e}")
#         return ""

# async def run_cumulative_agent_analysis(answers_list):
#     """
#     Send concatenated answers to `run_analysis_pipeline` and return structured breakdown:
#     returns tuple (per_agent_list, overall_avg)
#     per_agent_list: list of dicts [{'agent_name':'AgentHumility', 'score': 7}, ...]
#     """
#     if not answers_list:
#         return [], 0.0

#     transcript = "\n".join(answers_list)
#     try:
#         scores = await run_analysis_pipeline(transcript)  # expecting list of dicts
#     except Exception as e:
#         print(f"[run_cumulative_agent_analysis] run_analysis_pipeline failed: {e}")
#         scores = []

#     # Normalize and compute overall average
#     per_agent = []
#     total = 0.0
#     count = 0
#     for item in scores:
#         if isinstance(item, dict):
#             agent_name = item.get("agent_name", "Unknown")
#             score_val = item.get("score", 0)
#             # try to parse numeric scores
#             try:
#                 score_num = float(score_val)
#             except Exception:
#                 try:
#                     score_num = float(int(score_val))
#                 except Exception:
#                     score_num = 0.0
#             per_agent.append({"agent_name": agent_name, "score": score_num})
#             total += score_num
#             count += 1

#     overall_avg = (total / count) if count > 0 else 0.0
#     return per_agent, overall_avg

# # Use InterviewAnalyzer to keep per-turn detailed analyses for the final report
# async def analyze_and_store_turn(question, answer_text):
#     global analyzer
#     if analyzer is None:
#         analyzer = InterviewAnalyzer(candidate_name=getattr(session, "candidate_name", "Candidate"))
#     try:
#         # analyzer.analyze_response appends the analyzed turn to analyzer.analysis.turns internally
#         _ = await analyzer.analyze_response(question, answer_text)
#     except Exception as e:
#         print(f"[analyze_and_store_turn] error: {e}")

# # --- Gradio UI handlers ---

# def start_interview(name: str):
#     """Initialize session and analyzer."""
#     session.reset()
#     session.candidate_name = name.strip() or "Candidate"
#     session.turns = []  # ensure list exists
#     session.start_time = datetime.now().isoformat()
#     # Instantiate analyzer
#     global analyzer
#     analyzer = InterviewAnalyzer(candidate_name=session.candidate_name)
#     # initial UI states
#     first_q = QUESTIONS[0]
#     status_msg = f"Hello {session.candidate_name}. Starting interview."
#     return status_msg, first_q, gr.update(visible=True), gr.update(visible=True), ""

# async def submit_answer(audio_data):
#     """
#     Called when user clicks Submit Answer.
#     - transcribe
#     - append to session.turns
#     - run per-turn detailed analysis (stores into analyzer)
#     - run cumulative agent analysis across answers so far and return formatted live breakdown
#     """
#     if audio_data is None:
#         return "No audio received. Please record an answer.", "", "", gr.update(visible=True), gr.update(visible=True)

#     # transcribe
#     transcript = await transcribe(audio_data)

#     if transcript.strip() == "":
#         return "Could not transcribe audio. Try again.", "", "", gr.update(visible=True), gr.update(visible=True)

#     # store turn (simple Turn dataclass usage)
#     # idx = len(session.turns)
#     # turn = Turn(question=QUESTIONS[idx], audio_data=audio_data, transcript=transcript)
#     # session.turns.append(turn)
#     current_index = len(session.turns)

#     if current_index < len(QUESTIONS):
#         turn = Turn(
#             question=QUESTIONS[current_index],
#             audio_data=audio_data,
#             transcript=transcript
#         )
#         session.turns.append(turn)
#     else:
#         # All  questions answered — trigger final report generation
#         final_results = analyze_answers([t.transcript for t in session.turns])
#         pdf_path = generate_pdf_report(final_results)
#         return None, final_results, pdf_path



#     # Fire and forget per-turn detailed analyzer (but await to ensure stored)
#     await analyze_and_store_turn(turn.question, turn.transcript)

#     # Build answers list so far for cumulative analysis
#     answers_so_far = [t.transcript for t in session.turns]

#     per_agent, overall_avg = await run_cumulative_agent_analysis(answers_so_far)

#     # Format live progress text (markdown)
#     md = f"### Live Progress (after {len(answers_so_far)} answer(s))\n\n"
#     md += f"**Cumulative Overall Average:** **{overall_avg:.2f} / 10**\n\n"
#     md += "| Agent | Score |\n|---:|:---:|\n"
#     for p in per_agent:
#         md += f"| {p['agent_name']} | {p['score']:.2f} |\n"

#     # prepare next question or finish
#     if len(session.turns) < len(QUESTIONS):
#         next_q = QUESTIONS[len(session.turns)]
#         status = f"Recorded Q{len(session.turns)}. Moving to next question."
#         show_generate = False
#     else:
#         next_q = "All questions completed. Click 'Generate Final Report' to create PDF/HTML."
#         status = "Interview complete."
#         show_generate = True

#     return transcript, next_q, md, gr.update(visible=show_generate), gr.update(visible=show_generate)

# async def generate_final_report_and_pdf():
#     """
#     Create final HTML via analyzer.generate_report and convert to PDF.
#     Returns (html_str, html_path, pdf_path_or_none)
#     """
#     if analyzer is None or not analyzer.analysis.turns:
#         return "No interview data available. Please run the interview first.", None, None

#     # Make sure overall scores are computed
#     html = analyzer.generate_report(format="html")
#     html_path = report_generator.save_html_report(html, analyzer.candidate_name)
#     pdf_path = report_generator.create_pdf_report_from_html(html, analyzer.candidate_name)

#     return html, html_path, pdf_path

# # --- Gradio UI ---

# with gr.Blocks(title="🎙️ Interview + Live Cumulative Analysis + PDF Report") as demo:
#     gr.Markdown("## Interview Assistant — live per-agent breakdown and final PDF report")

#     with gr.Row():
#         with gr.Column(scale=1):
#             name_input = gr.Textbox(label="Candidate Name", placeholder="Enter candidate name")
#             start_btn = gr.Button("Start Interview")

#             question_display = gr.Textbox(label="Current Question", interactive=False, lines=3)
#             audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record your answer", visible=False)
#             submit_btn = gr.Button("Submit Answer", visible=False)
#             generate_report_btn = gr.Button("Generate Final Report (HTML + PDF)", visible=False)

#             status = gr.Markdown("")

#         with gr.Column(scale=1):
#             transcript_box = gr.Textbox(label="Last Transcript", lines=4, interactive=False)
#             live_progress = gr.Markdown("### Live progress will appear here")
#             final_report_html = gr.HTML("")
#             download_html = gr.File(label="Download HTML report")
#             download_pdf = gr.File(label="Download PDF report")

#     # Start interview
#     start_btn.click(
#         fn=start_interview,
#         inputs=[name_input],
#         outputs=[status, question_display, audio_input, submit_btn, live_progress]
#     )

#     # When user clicks Submit Answer
#     submit_btn.click(
#         fn=submit_answer,
#         inputs=[audio_input],
#         outputs=[transcript_box, question_display, live_progress, generate_report_btn, generate_report_btn]
#     )

#     # Generate final report
#     generate_report_btn.click(
#         fn=generate_final_report_and_pdf,
#         inputs=None,
#         outputs=[final_report_html, download_html, download_pdf]
#     )

# if __name__ == "__main__":
#     demo.launch(server_name="localhost", server_port=7860)























##chatgpt 5pro 3rd 
# interview_gradio_app.py
import os
import numpy as np
import torch
import gradio as gr
import librosa
from faster_whisper import WhisperModel
from datetime import datetime

from interview_state import SessionState, Turn
from interview_analyzer import InterviewAnalyzer
from backend.agent_manager import run_analysis_pipeline
from report_generator import build_full_report_html, save_html_report, create_pdf_report_from_html

os.makedirs("reports", exist_ok=True)

QUESTIONS = [
    "1) Can you tell me about a time when you received constructive criticism? How did you handle it?",
    "2) Describe a situation where you had to work with a difficult team member. How did you handle it?",
    "3) Tell me about a time when you made a mistake at work. How did you address it?",
    "4) How do you handle situations where you need to learn something new?",
    "5) Can you share an example of when you had to adapt to a significant change at work?"
]

MODEL_SIZE = "base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8")

session = SessionState()
analyzer: InterviewAnalyzer | None = None

def preprocess_audio(audio_data):
    if audio_data is None:
        return np.zeros((0,), dtype=np.float32)
    sr, arr = audio_data
    if arr is None:
        return np.zeros((0,), dtype=np.float32)
    if arr.ndim > 1 and arr.shape[1] == 2:
        arr = arr.mean(axis=1)
    if arr.dtype != np.float32:
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
        else:
            arr = arr.astype(np.float32)
    if sr != 16000:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
    return arr

async def transcribe(audio_data):
    audio = preprocess_audio(audio_data)
    try:
        segments, _ = whisper_model.transcribe(
            audio, language="en", beam_size=5, vad_filter=True
        )
        return " ".join(seg.text for seg in segments).strip()
    except Exception as e:
        print(f"[transcribe] error: {e}")
        return ""

async def run_cumulative_agent_analysis(answers_list):
    """
    Returns per_agent list from LLM agents and overall average.
    Handles AgentScore dataclass or dict.
    """
    if not answers_list:
        return [], 0.0

    transcript = "\n".join(answers_list)
    try:
        scores = await run_analysis_pipeline(transcript)
    except Exception as e:
        print(f"[run_cumulative_agent_analysis] failed: {e}")
        scores = []

    per_agent, total, count = [], 0.0, 0
    for item in scores:
        # item could be a dataclass or dict
        try:
            agent_name = getattr(item, "agent_name", None) or item.get("agent_name", "Unknown")
            score_val = getattr(item, "score", None) if hasattr(item, "score") else item.get("score", 0)
            evidence = getattr(item, "evidence", None) if hasattr(item, "evidence") else item.get("evidence", "")
        except Exception:
            agent_name, score_val, evidence = "Unknown", 0, ""

        try:
            score_num = float(score_val)
        except Exception:
            try:
                score_num = float(int(score_val))
            except Exception:
                score_num = 0.0

        per_agent.append({"agent_name": agent_name, "score": score_num, "evidence": evidence})
        total += score_num
        count += 1

    overall_avg = (total / count) if count > 0 else 0.0
    return per_agent, overall_avg

async def analyze_and_store_turn(question, answer_text):
    global analyzer
    if analyzer is None:
        analyzer = InterviewAnalyzer(candidate_name=getattr(session, "candidate_name", "Candidate"))
    try:
        await analyzer.analyze_response(question, answer_text)
    except Exception as e:
        print(f"[analyze_and_store_turn] error: {e}")

def start_interview(name: str):
    session.reset()
    session.candidate_name = name.strip() or "Candidate"
    session.turns = []
    session.start_time = datetime.now().isoformat()

    global analyzer
    analyzer = InterviewAnalyzer(candidate_name=session.candidate_name)

    first_q = QUESTIONS[0]
    status_msg = f"Hello {session.candidate_name}. Starting interview."
    return status_msg, first_q, gr.update(visible=True), gr.update(visible=True), ""

def _format_live_md(per_agent, overall_avg):
    md = f"### Live Progress\n\n**LLM Agents Overall Avg:** **{overall_avg:.2f} / 10**\n\n"
    md += "| Agent | Score |\n|---:|:---:|\n"
    for p in per_agent:
        md += f"| {p['agent_name']} | {p['score']:.2f} |\n"
    return md

# def _format_final_table_md(llm_agents, analyzer_payload):
#     # LLM agents
#     md = "## Final Agentic Analysis Summary\n\n"
#     md += "### LLM Agents\n\n"
#     md += "| Agent | Score | Evidence |\n|---|:---:|---|\n"
#     for a in llm_agents:
#         ev = (a.get("evidence") or "").replace("\n"," ").strip()
#         md += f"| {a.get('agent_name','')} | {a.get('score',0):.2f} | {ev[:180]}{'…' if len(ev)>180 else ''} |\n"

#     # Four HR traits (averages)
#     ov = analyzer_payload["overall_scores"]
#     md += "\n\n### Core Behavioral Traits (Average)\n\n"
#     md += "| Trait | Score |\n|---|:---:|\n"
#     for k in ["humility","learning","feedback","mistakes"]:
#         md += f"| {k.capitalize()} | {ov.get(k,0):.1f} |\n"

#     # suggestions
#     tips = analyzer_payload["summary_suggestions"]
#     md += "\n\n### Suggestions\n\n"
#     for s in tips:
#         md += f"- {s}\n"
#     return md

# ---- replace this whole function in interview_gradio_app.py ----
def _clean_agent_name(name: str) -> str:
    if not name:
        return ""
    return name[:-5] if name.endswith("Agent") else name

def _format_final_table_md(llm_agents, analyzer_payload):
    """
    Builds the final on-screen markdown summary.

    - Cleans agent names (drop trailing 'Agent')
    - Shows only Humility in 'Core Behavioral Traits'
    - Does not assume 'summary_suggestions' exists
    """
    # 1) LLM agents table
    md = "## Final Agentic Analysis Summary\n\n"
    md += "### LLM Agents\n\n"
    md += "| Agent | Score | Evidence |\n|---|:---:|---|\n"
    for a in llm_agents:
        agent_name = _clean_agent_name(a.get("agent_name", ""))
        score = a.get("score", 0.0)
        evidence = (a.get("evidence") or "").replace("\n", " ").strip()
        md += f"| {agent_name} | {float(score):.2f} | {evidence[:180]}{'…' if len(evidence) > 180 else ''} |\n"

    # 2) Core Behavioral Traits (UI) — show ONLY Humility
    ov = analyzer_payload.get("overall_scores", {}) or {}
    hum = ov.get("humility", None)
    md += "\n\n### Core Behavioral Traits (Summary)\n\n"
    md += "| Trait | Score |\n|---|:---:|\n"
    if hum is not None:
        md += f"| Humility | {float(hum):.1f} |\n"
    else:
        md += "| Humility | — |\n"

    # 3) Suggestions (optional): only include if present
    tips = analyzer_payload.get("summary_suggestions", []) or []
    if tips:
        md += "\n\n### Suggestions\n\n"
        for s in tips:
            md += f"- {s}\n"

    return md


async def submit_answer(audio_data):
    if audio_data is None:
        return "No audio received. Please record an answer.", "", "", gr.update(visible=True), gr.update(visible=True)

    transcript = await transcribe(audio_data)
    if transcript.strip() == "":
        return "Could not transcribe audio. Try again.", "", "", gr.update(visible=True), gr.update(visible=True)

    current_index = len(session.turns)
    if current_index < len(QUESTIONS):
        turn = Turn(question=QUESTIONS[current_index], audio_data=audio_data, transcript=transcript)
        session.turns.append(turn)
    else:
        # Safety: should not happen
        pass

    await analyze_and_store_turn(session.turns[-1].question, session.turns[-1].transcript)

    # Build answers list so far for LLM agents
    answers_so_far = [t.transcript for t in session.turns]
    per_agent, overall_avg = await run_cumulative_agent_analysis(answers_so_far)

    # If finished, show final full summary table (+ suggestions) and enable report button
    if len(session.turns) == len(QUESTIONS):
        payload = analyzer.to_summary_payload()
        final_md = _format_final_table_md(per_agent, payload)
        next_q = "All questions completed."
        status = "Interview complete. You can now generate the final PDF report."
        show_generate = True
        return transcript, next_q, final_md, gr.update(visible=show_generate), gr.update(visible=show_generate)

    # Otherwise, normal live progress
    next_q = QUESTIONS[len(session.turns)]
    status = f"Recorded Q{len(session.turns)}. Moving to next question."
    show_generate = False
    live_md = _format_live_md(per_agent, overall_avg)
    return transcript, next_q, live_md, gr.update(visible=show_generate), gr.update(visible=show_generate)

async def generate_final_report_and_pdf():
    """
    Builds HTML from analyzer + LLM agents, saves HTML, tries to create PDF.
    Returns (html_str, html_path, pdf_path_or_none)
    """
    if analyzer is None or not analyzer.analysis.turns:
        return "No interview data available. Please run the interview first.", None, None

    # Collect LLM agent results on all answers (final)
    answers_all = [t.transcript for t in session.turns]
    llm_agents, _ = await run_cumulative_agent_analysis(answers_all)

    # Build final HTML from payload
    payload = analyzer.to_summary_payload()
    html = build_full_report_html(payload, llm_agents=llm_agents)
    html_path = save_html_report(html, analyzer.candidate_name)
    pdf_path = create_pdf_report_from_html(html, analyzer.candidate_name)

    if pdf_path is None:
        # WeasyPrint not installed; still return HTML and tell user
        html += "<!-- PDF generation unavailable (WeasyPrint not installed). -->"

    return html, html_path, (pdf_path or None)

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

    start_btn.click(
        fn=start_interview,
        inputs=[name_input],
        outputs=[status, question_display, audio_input, submit_btn, live_progress]
    )

    submit_btn.click(
        fn=submit_answer,
        inputs=[audio_input],
        outputs=[transcript_box, question_display, live_progress, generate_report_btn, generate_report_btn]
    )

    generate_report_btn.click(
        fn=generate_final_report_and_pdf,
        inputs=None,
        outputs=[final_report_html, download_html, download_pdf]
    )

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860)
