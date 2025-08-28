

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























# ##chatgpt 5pro 3rd 
# # interview_gradio_app.py
# import os
# import numpy as np
# import torch
# import gradio as gr
# import librosa
# from faster_whisper import WhisperModel
# from datetime import datetime

# from interview_state import SessionState, Turn
# from interview_analyzer import InterviewAnalyzer
# from backend.agent_manager import run_analysis_pipeline
# from report_generator import build_full_report_html, save_html_report, create_pdf_report_from_html

# os.makedirs("reports", exist_ok=True)

# QUESTIONS = [
#     "1) Can you tell me about a time when you received constructive criticism? How did you handle it?",
#     "2) Describe a situation where you had to work with a difficult team member. How did you handle it?",
#     "3) Tell me about a time when you made a mistake at work. How did you address it?",
#     "4) How do you handle situations where you need to learn something new?",
#     "5) Can you share an example of when you had to adapt to a significant change at work?"
# ]

# MODEL_SIZE = "base"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8")

# session = SessionState()
# analyzer: InterviewAnalyzer | None = None

# def preprocess_audio(audio_data):
#     if audio_data is None:
#         return np.zeros((0,), dtype=np.float32)
#     sr, arr = audio_data
#     if arr is None:
#         return np.zeros((0,), dtype=np.float32)
#     if arr.ndim > 1 and arr.shape[1] == 2:
#         arr = arr.mean(axis=1)
#     if arr.dtype != np.float32:
#         if np.issubdtype(arr.dtype, np.integer):
#             arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
#         else:
#             arr = arr.astype(np.float32)
#     if sr != 16000:
#         arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
#     return arr

# async def transcribe(audio_data):
#     audio = preprocess_audio(audio_data)
#     try:
#         segments, _ = whisper_model.transcribe(
#             audio, language="en", beam_size=5, vad_filter=True
#         )
#         return " ".join(seg.text for seg in segments).strip()
#     except Exception as e:
#         print(f"[transcribe] error: {e}")
#         return ""

# async def run_cumulative_agent_analysis(answers_list):
#     """
#     Returns per_agent list from LLM agents and overall average.
#     Handles AgentScore dataclass or dict.
#     """
#     if not answers_list:
#         return [], 0.0

#     transcript = "\n".join(answers_list)
#     try:
#         scores = await run_analysis_pipeline(transcript)
#     except Exception as e:
#         print(f"[run_cumulative_agent_analysis] failed: {e}")
#         scores = []

#     per_agent, total, count = [], 0.0, 0
#     for item in scores:
#         # item could be a dataclass or dict
#         try:
#             agent_name = getattr(item, "agent_name", None) or item.get("agent_name", "Unknown")
#             score_val = getattr(item, "score", None) if hasattr(item, "score") else item.get("score", 0)
#             evidence = getattr(item, "evidence", None) if hasattr(item, "evidence") else item.get("evidence", "")
#         except Exception:
#             agent_name, score_val, evidence = "Unknown", 0, ""

#         try:
#             score_num = float(score_val)
#         except Exception:
#             try:
#                 score_num = float(int(score_val))
#             except Exception:
#                 score_num = 0.0

#         per_agent.append({"agent_name": agent_name, "score": score_num, "evidence": evidence})
#         total += score_num
#         count += 1

#     overall_avg = (total / count) if count > 0 else 0.0
#     return per_agent, overall_avg

# async def analyze_and_store_turn(question, answer_text):
#     global analyzer
#     if analyzer is None:
#         analyzer = InterviewAnalyzer(candidate_name=getattr(session, "candidate_name", "Candidate"))
#     try:
#         await analyzer.analyze_response(question, answer_text)
#     except Exception as e:
#         print(f"[analyze_and_store_turn] error: {e}")

# def start_interview(name: str):
#     session.reset()
#     session.candidate_name = name.strip() or "Candidate"
#     session.turns = []
#     session.start_time = datetime.now().isoformat()

#     global analyzer
#     analyzer = InterviewAnalyzer(candidate_name=session.candidate_name)

#     first_q = QUESTIONS[0]
#     status_msg = f"Hello {session.candidate_name}. Starting interview."
#     return status_msg, first_q, gr.update(visible=True), gr.update(visible=True), ""

# def _format_live_md(per_agent, overall_avg):
#     md = f"### Live Progress\n\n**LLM Agents Overall Avg:** **{overall_avg:.2f} / 10**\n\n"
#     md += "| Agent | Score |\n|---:|:---:|\n"
#     for p in per_agent:
#         md += f"| {p['agent_name']} | {p['score']:.2f} |\n"
#     return md

# # def _format_final_table_md(llm_agents, analyzer_payload):
# #     # LLM agents
# #     md = "## Final Agentic Analysis Summary\n\n"
# #     md += "### LLM Agents\n\n"
# #     md += "| Agent | Score | Evidence |\n|---|:---:|---|\n"
# #     for a in llm_agents:
# #         ev = (a.get("evidence") or "").replace("\n"," ").strip()
# #         md += f"| {a.get('agent_name','')} | {a.get('score',0):.2f} | {ev[:180]}{'…' if len(ev)>180 else ''} |\n"

# #     # Four HR traits (averages)
# #     ov = analyzer_payload["overall_scores"]
# #     md += "\n\n### Core Behavioral Traits (Average)\n\n"
# #     md += "| Trait | Score |\n|---|:---:|\n"
# #     for k in ["humility","learning","feedback","mistakes"]:
# #         md += f"| {k.capitalize()} | {ov.get(k,0):.1f} |\n"

# #     # suggestions
# #     tips = analyzer_payload["summary_suggestions"]
# #     md += "\n\n### Suggestions\n\n"
# #     for s in tips:
# #         md += f"- {s}\n"
# #     return md

# # ---- replace this whole function in interview_gradio_app.py ----
# def _clean_agent_name(name: str) -> str:
#     if not name:
#         return ""
#     return name[:-5] if name.endswith("Agent") else name

# def _format_final_table_md(llm_agents, analyzer_payload):
#     """
#     Builds the final on-screen markdown summary.

#     - Cleans agent names (drop trailing 'Agent')
#     - Shows only Humility in 'Core Behavioral Traits'
#     - Does not assume 'summary_suggestions' exists
#     """
#     # 1) LLM agents table
#     md = "## Final Agentic Analysis Summary\n\n"
#     md += "### LLM Agents\n\n"
#     md += "| Agent | Score | Evidence |\n|---|:---:|---|\n"
#     for a in llm_agents:
#         agent_name = _clean_agent_name(a.get("agent_name", ""))
#         score = a.get("score", 0.0)
#         evidence = (a.get("evidence") or "").replace("\n", " ").strip()
#         md += f"| {agent_name} | {float(score):.2f} | {evidence[:180]}{'…' if len(evidence) > 180 else ''} |\n"

#     # 2) Core Behavioral Traits (UI) — show ONLY Humility
#     ov = analyzer_payload.get("overall_scores", {}) or {}
#     hum = ov.get("humility", None)
#     md += "\n\n### Core Behavioral Traits (Summary)\n\n"
#     md += "| Trait | Score |\n|---|:---:|\n"
#     if hum is not None:
#         md += f"| Humility | {float(hum):.1f} |\n"
#     else:
#         md += "| Humility | — |\n"

#     # 3) Suggestions (optional): only include if present
#     tips = analyzer_payload.get("summary_suggestions", []) or []
#     if tips:
#         md += "\n\n### Suggestions\n\n"
#         for s in tips:
#             md += f"- {s}\n"

#     return md


# async def submit_answer(audio_data):
#     if audio_data is None:
#         return "No audio received. Please record an answer.", "", "", gr.update(visible=True), gr.update(visible=True)

#     transcript = await transcribe(audio_data)
#     if transcript.strip() == "":
#         return "Could not transcribe audio. Try again.", "", "", gr.update(visible=True), gr.update(visible=True)

#     current_index = len(session.turns)
#     if current_index < len(QUESTIONS):
#         turn = Turn(question=QUESTIONS[current_index], audio_data=audio_data, transcript=transcript)
#         session.turns.append(turn)
#     else:
#         # Safety: should not happen
#         pass

#     await analyze_and_store_turn(session.turns[-1].question, session.turns[-1].transcript)

#     # Build answers list so far for LLM agents
#     answers_so_far = [t.transcript for t in session.turns]
#     per_agent, overall_avg = await run_cumulative_agent_analysis(answers_so_far)

#     # If finished, show final full summary table (+ suggestions) and enable report button
#     if len(session.turns) == len(QUESTIONS):
#         payload = analyzer.to_summary_payload()
#         final_md = _format_final_table_md(per_agent, payload)
#         next_q = "All questions completed."
#         status = "Interview complete. You can now generate the final PDF report."
#         show_generate = True
#         return transcript, next_q, final_md, gr.update(visible=show_generate), gr.update(visible=show_generate)

#     # Otherwise, normal live progress
#     next_q = QUESTIONS[len(session.turns)]
#     status = f"Recorded Q{len(session.turns)}. Moving to next question."
#     show_generate = False
#     live_md = _format_live_md(per_agent, overall_avg)
#     return transcript, next_q, live_md, gr.update(visible=show_generate), gr.update(visible=show_generate)

# async def generate_final_report_and_pdf():
#     """
#     Builds HTML from analyzer + LLM agents, saves HTML, tries to create PDF.
#     Returns (html_str, html_path, pdf_path_or_none)
#     """
#     if analyzer is None or not analyzer.analysis.turns:
#         return "No interview data available. Please run the interview first.", None, None

#     # Collect LLM agent results on all answers (final)
#     answers_all = [t.transcript for t in session.turns]
#     llm_agents, _ = await run_cumulative_agent_analysis(answers_all)

#     # Build final HTML from payload
#     payload = analyzer.to_summary_payload()
#     html = build_full_report_html(payload, llm_agents=llm_agents)
#     html_path = save_html_report(html, analyzer.candidate_name)
#     pdf_path = create_pdf_report_from_html(html, analyzer.candidate_name)

#     if pdf_path is None:
#         # WeasyPrint not installed; still return HTML and tell user
#         html += "<!-- PDF generation unavailable (WeasyPrint not installed). -->"

#     return html, html_path, (pdf_path or None)

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

#     start_btn.click(
#         fn=start_interview,
#         inputs=[name_input],
#         outputs=[status, question_display, audio_input, submit_btn, live_progress]
#     )

#     submit_btn.click(
#         fn=submit_answer,
#         inputs=[audio_input],
#         outputs=[transcript_box, question_display, live_progress, generate_report_btn, generate_report_btn]
#     )

#     generate_report_btn.click(
#         fn=generate_final_report_and_pdf,
#         inputs=None,
#         outputs=[final_report_html, download_html, download_pdf]
#     )

# if __name__ == "__main__":
#     demo.launch(server_name="localhost", server_port=7860)












# ##chatgpt 5pro 4th 
# # interview_gradio_app.py
# import os
# import numpy as np
# import torch
# import gradio as gr
# import librosa
# from faster_whisper import WhisperModel
# from datetime import datetime

# from interview_state import SessionState, Turn
# from interview_analyzer import InterviewAnalyzer
# from backend.agent_manager import run_analysis_pipeline
# from report_generator import build_full_report_html, save_html_report, create_pdf_report_from_html

# os.makedirs("reports", exist_ok=True)

# QUESTIONS = [
#     "1) Can you tell me about a time when you received constructive criticism? How did you handle it?",
#     "2) Describe a situation where you had to work with a difficult team member. How did you handle it?",
#     "3) Tell me about a time when you made a mistake at work. How did you address it?",
#     "4) How do you handle situations where you need to learn something new?",
#     "5) Can you share an example of when you had to adapt to a significant change at work?"
# ]

# MODEL_SIZE = "base"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8")

# session = SessionState()
# analyzer: InterviewAnalyzer | None = None

# def preprocess_audio(audio_data):
#     if audio_data is None:
#         return np.zeros((0,), dtype=np.float32)
#     sr, arr = audio_data
#     if arr is None:
#         return np.zeros((0,), dtype=np.float32)
#     if arr.ndim > 1 and arr.shape[1] == 2:
#         arr = arr.mean(axis=1)
#     if arr.dtype != np.float32:
#         if np.issubdtype(arr.dtype, np.integer):
#             arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
#         else:
#             arr = arr.astype(np.float32)
#     if sr != 16000:
#         arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
#     return arr

# async def transcribe(audio_data):
#     audio = preprocess_audio(audio_data)
#     try:
#         segments, _ = whisper_model.transcribe(
#             audio, language="en", beam_size=5, vad_filter=True
#         )
#         return " ".join(seg.text for seg in segments).strip()
#     except Exception as e:
#         print(f"[transcribe] error: {e}")
#         return ""

# async def run_cumulative_agent_analysis(answers_list):
#     """
#     Returns per_agent list from LLM agents and overall average.
#     Handles AgentScore dataclass or dict.
#     """
#     if not answers_list:
#         return [], 0.0

#     transcript = "\n".join(answers_list)
#     try:
#         scores = await run_analysis_pipeline(transcript)
#     except Exception as e:
#         print(f"[run_cumulative_agent_analysis] failed: {e}")
#         scores = []

#     per_agent, total, count = [], 0.0, 0
#     for item in scores:
#         # item could be a dataclass or dict
#         try:
#             agent_name = getattr(item, "agent_name", None) or item.get("agent_name", "Unknown")
#             score_val = getattr(item, "score", None) if hasattr(item, "score") else item.get("score", 0)
#             evidence = getattr(item, "evidence", None) if hasattr(item, "evidence") else item.get("evidence", "")
#         except Exception:
#             agent_name, score_val, evidence = "Unknown", 0, ""

#         try:
#             score_num = float(score_val)
#         except Exception:
#             try:
#                 score_num = float(int(score_val))
#             except Exception:
#                 score_num = 0.0

#         per_agent.append({"agent_name": agent_name, "score": score_num, "evidence": evidence})
#         total += score_num
#         count += 1

#     overall_avg = (total / count) if count > 0 else 0.0
#     return per_agent, overall_avg

# async def analyze_and_store_turn(question, answer_text):
#     global analyzer
#     if analyzer is None:
#         analyzer = InterviewAnalyzer(candidate_name=getattr(session, "candidate_name", "Candidate"))
#     try:
#         await analyzer.analyze_response(question, answer_text)
#     except Exception as e:
#         print(f"[analyze_and_store_turn] error: {e}")

# def start_interview(name: str):
#     session.reset()
#     session.candidate_name = name.strip() or "Candidate"
#     session.turns = []
#     session.start_time = datetime.now().isoformat()

#     global analyzer
#     analyzer = InterviewAnalyzer(candidate_name=session.candidate_name)

#     first_q = QUESTIONS[0]
#     status_msg = f"Hello {session.candidate_name}. Starting interview."
#     return status_msg, first_q, gr.update(visible=True), gr.update(visible=True), ""
# ##tak4
# # def _format_live_md(per_agent, _overall_avg_not_used):
# #     # filter out Pronoun*/IDontKnow for display AND for the average
# #     shown = [a for a in per_agent if not _hide_in_live_table(a.get("agent_name", ""))]
# #     if shown:
# #         avg = sum(float(a.get("score", 0.0)) for a in shown) / len(shown)
# #     else:
# #         avg = 0.0

# #     md = f"### Live Progress\n\n**LLM Agents Avg (filtered):** **{avg:.2f} / 10**\n\n"
# #     md += "| Agent | Score |\n|---|:---:|\n"
# #     for a in shown:
# #         md += f"| {_clean_agent_name(a.get('agent_name',''))} | {float(a.get('score',0.0)):.2f} |\n"
# #     return md

# ##take 5
# def _format_live_md(per_agent, _overall_avg_not_used):
#     shown = [a for a in per_agent if not _hide_in_live_table(a.get("agent_name",""))]
#     live_avg = (sum(float(a.get("score",0.0)) for a in shown) / len(shown)) if shown else 0.0

#     md = f"### Live Progress\n\n**LLM Agents Avg (filtered):** **{live_avg:.2f} / 10**\n\n"
#     md += "| Agent | Score |\n|---|:---:|\n"
#     for a in shown:
#         md += f"| {_clean_agent_name(a.get('agent_name',''))} | {float(a.get('score',0.0)):.2f} |\n"
#     return md



# # def _format_final_table_md(llm_agents, analyzer_payload):
# #     # LLM agents
# #     md = "## Final Agentic Analysis Summary\n\n"
# #     md += "### LLM Agents\n\n"
# #     md += "| Agent | Score | Evidence |\n|---|:---:|---|\n"
# #     for a in llm_agents:
# #         ev = (a.get("evidence") or "").replace("\n"," ").strip()
# #         md += f"| {a.get('agent_name','')} | {a.get('score',0):.2f} | {ev[:180]}{'…' if len(ev)>180 else ''} |\n"

# #     # Four HR traits (averages)
# #     ov = analyzer_payload["overall_scores"]
# #     md += "\n\n### Core Behavioral Traits (Average)\n\n"
# #     md += "| Trait | Score |\n|---|:---:|\n"
# #     for k in ["humility","learning","feedback","mistakes"]:
# #         md += f"| {k.capitalize()} | {ov.get(k,0):.1f} |\n"

# #     # suggestions
# #     tips = analyzer_payload["summary_suggestions"]
# #     md += "\n\n### Suggestions\n\n"
# #     for s in tips:
# #         md += f"- {s}\n"
# #     return md

# # ---- replace this whole function in interview_gradio_app.py ----
# # --- helpers for agent filtering / display ---

# # --- Agent display / selection helpers ---

# EIGHT_HUMILITY_AGENTS = {
#     "admitmistake", "mindchange", "learnermindset",
#     "bragflag", "blameshift", "knowitall",
#     "feedbackacceptance", "supportgrowth"
# }

# def _base_name(name: str) -> str:
#     n = (name or "").lower()
#     return n[:-5] if n.endswith("agent") else n

# def _clean_agent_name(name: str) -> str:
#     if not name: return ""
#     return name[:-5] if name.endswith("Agent") else name

# def _hide_in_live_table(name: str) -> bool:
#     n = (name or "").lower()
#     return ("pronoun" in n) or ("idontknow" in n)

# def _final_humility_from_llm(llm_agents: list[dict]) -> float:
#     vals = []
#     for a in llm_agents or []:
#         base = _base_name(a.get("agent_name",""))
#         if base in EIGHT_HUMILITY_AGENTS:
#             try:
#                 vals.append(float(a.get("score", 0.0)))
#             except Exception:
#                 pass
#     return round(sum(vals) / len(vals), 1) if vals else 0.0


# def _exclude_agent_name(name: str) -> bool:
#     n = (name or "").lower()
#     return any(key in n for key in ["idontknow", "pronoun", "sharecredit", "praisehandling", "precisehandling"])


# def _is_anti_humility(name: str) -> bool:
#     n = (name or "").lower()
#     return any(k in n for k in ["bragflag", "knowitall", "blameshift"])

# def _agent_only_humility(llm_agents: list[dict]) -> float:
#     vals = []
#     for a in llm_agents or []:
#         name = str(a.get("agent_name", ""))
#         if _exclude_agent_name(name):
#             continue
#         try:
#             s = float(a.get("score", 0.0))
#         except Exception:
#             s = 0.0
#         s = max(0.0, min(10.0, s))
#         if _is_anti_humility(name):
#             s = 10.0 - s
#         vals.append(s)
#     return round(sum(vals) / len(vals), 1) if vals else 0.0

# ##take 4
# # def _format_final_table_md(llm_agents, analyzer_payload):
# #     """
# #     Final on-screen summary:
# #       - LLM Agents table: hide Pronoun*, IDontKnow, ShareCredit, PraiseHandling
# #       - Core summary: show ONLY Humility computed from included agents
# #     """
# #     # LLM table (filtered + cleaned names)
# #     md = "## Final Agentic Analysis Summary\n\n"
# #     md += "### LLM Agents\n\n"
# #     md += "| Agent | Score | Evidence |\n|---|:---:|---|\n"
# #     for a in llm_agents:
# #         name = str(a.get("agent_name",""))
# #         if _exclude_agent_name(name):
# #             continue
# #         score = float(a.get("score", 0.0))
# #         ev = (a.get("evidence") or "").replace("\n", " ").strip()
# #         md += f"| {_clean_agent_name(name)} | {score:.2f} | {ev[:180]}{'…' if len(ev)>180 else ''} |\n"

# #     # Core (Humility only) from agents
# #     hum = _agent_only_humility(llm_agents)
# #     md += "\n\n### Core Behavioral Traits (Summary)\n\n"
# #     md += "| Trait | Score |\n|---|:---:|\n"
# #     md += f"| Humility | {hum:.1f} |\n"

# #     return md
# def _format_final_table_md(llm_agents, analyzer_payload):
#     md = "## Final Agentic Analysis Summary\n\n"
#     md += "### LLM Agents\n\n"
#     md += "| Agent | Score | Evidence |\n|---|:---:|---|\n"
#     for a in llm_agents:
#         name = str(a.get("agent_name",""))
#         # keep table cleaner (optional): hide pronoun/idontknow/praisehandling like PDF
#         if any(k in name.lower() for k in ["pronoun","idontknow","praisehandling","precisehandling"]):
#             continue
#         score = float(a.get("score", 0.0))
#         ev = (a.get("evidence") or "").replace("\n", " ").strip()
#         md += f"| {_clean_agent_name(name)} | {score:.2f} | {ev[:180]}{'…' if len(ev)>180 else ''} |\n"

#     hum = _final_humility_from_llm(llm_agents)
#     md += "\n\n### Core Behavioral Traits (Summary)\n\n"
#     md += "| Trait | Score |\n|---|:---:|\n"
#     md += f"| Humility | {hum:.1f} |\n"
#     return md


# async def submit_answer(audio_data):
#     if audio_data is None:
#         return "No audio received. Please record an answer.", "", "", gr.update(visible=True), gr.update(visible=True)

#     transcript = await transcribe(audio_data)
#     if transcript.strip() == "":
#         return "Could not transcribe audio. Try again.", "", "", gr.update(visible=True), gr.update(visible=True)

#     current_index = len(session.turns)
#     if current_index < len(QUESTIONS):
#         turn = Turn(question=QUESTIONS[current_index], audio_data=audio_data, transcript=transcript)
#         session.turns.append(turn)
#     else:
#         # Safety: should not happen
#         pass

#     await analyze_and_store_turn(session.turns[-1].question, session.turns[-1].transcript)

#     # Build answers list so far for LLM agents
#     answers_so_far = [t.transcript for t in session.turns]
#     per_agent, overall_avg = await run_cumulative_agent_analysis(answers_so_far)

#     # If finished, show final full summary table (+ suggestions) and enable report button
#     if len(session.turns) == len(QUESTIONS):
#         payload = analyzer.to_summary_payload()
#         final_md = _format_final_table_md(per_agent, payload)
#         next_q = "All questions completed."
#         status = "Interview complete. You can now generate the final PDF report."
#         show_generate = True
#         return transcript, next_q, final_md, gr.update(visible=show_generate), gr.update(visible=show_generate)

#     # Otherwise, normal live progress
#     next_q = QUESTIONS[len(session.turns)]
#     status = f"Recorded Q{len(session.turns)}. Moving to next question."
#     show_generate = False
#     live_md = _format_live_md(per_agent, overall_avg)
#     return transcript, next_q, live_md, gr.update(visible=show_generate), gr.update(visible=show_generate)

# async def generate_final_report_and_pdf():
#     """
#     Builds HTML from analyzer + LLM agents, saves HTML, tries to create PDF.
#     Returns (html_str, html_path, pdf_path_or_none)
#     """
#     if analyzer is None or not analyzer.analysis.turns:
#         return "No interview data available. Please run the interview first.", None, None

#     # Collect LLM agent results on all answers (final)
#     answers_all = [t.transcript for t in session.turns]
#     llm_agents, _ = await run_cumulative_agent_analysis(answers_all)

#     # Build final HTML from payload
#     payload = analyzer.to_summary_payload()
#     html = build_full_report_html(payload, llm_agents=llm_agents)
#     html_path = save_html_report(html, analyzer.candidate_name)
#     pdf_path = create_pdf_report_from_html(html, analyzer.candidate_name)

#     if pdf_path is None:
#         # WeasyPrint not installed; still return HTML and tell user
#         html += "<!-- PDF generation unavailable (WeasyPrint not installed). -->"

#     return html, html_path, (pdf_path or None)

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

#     start_btn.click(
#         fn=start_interview,
#         inputs=[name_input],
#         outputs=[status, question_display, audio_input, submit_btn, live_progress]
#     )

#     submit_btn.click(
#         fn=submit_answer,
#         inputs=[audio_input],
#         outputs=[transcript_box, question_display, live_progress, generate_report_btn, generate_report_btn]
#     )

#     generate_report_btn.click(
#         fn=generate_final_report_and_pdf,
#         inputs=None,
#         outputs=[final_report_html, download_html, download_pdf]
#     )

# if __name__ == "__main__":
#     demo.launch(server_name="localhost", server_port=7860, share=True)
















# ##chatgpt-5 - 6 th
# # interview_gradio_app.py
# import os
# import numpy as np
# import torch
# import gradio as gr
# import librosa
# import asyncio
# from faster_whisper import WhisperModel
# from datetime import datetime
# from io import BytesIO

# from interview_state import SessionState, Turn
# from interview_analyzer import InterviewAnalyzer
# from backend.agent_manager import run_analysis_pipeline
# import report_generator  # your current generator (unchanged)

# def synth_tts_eleven(text: str, voice_id: str | None = None) -> tuple[int, np.ndarray] | None:
#     """
#     Return (sr, np.float32 mono) from ElevenLabs TTS, or None if unavailable.
#     Requires: pip install elevenlabs ; env ELEVEN_API_KEY
#     """
#     try:
#         from elevenlabs import VoiceSettings
#         from elevenlabs.client import ElevenLabs
#         client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY", ""))
#         if not client.api_key:
#             return None
#         voice = voice_id or os.getenv("ELEVEN_VOICE_ID", "Rachel")
#         # v2 tts endpoint → bytes (mp3 or wav). We'll request PCM.
#         audio = client.text_to_speech.convert_as_stream(
#             voice_id=voice,
#             optimize_streaming_latency="0",
#             output_format="pcm_16000",
#             text=text,
#             voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.7),
#         )
#         # stream yields chunks of bytes; concatenate to bytes
#         pcm_bytes = b"".join(chunk for chunk in audio)
#         # convert 16k PCM 16-bit little-endian to float32 mono
#         import numpy as np
#         arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
#         return 16000, arr
#     except Exception as e:
#         print("[TTS] ElevenLabs unavailable or failed:", e)
#         return None


# # -------------------- CONFIG --------------------
# os.makedirs("reports", exist_ok=True)
# DEFAULT_QUESTIONS = [
#     "1) Can you tell me about a time when you received constructive criticism? How did you handle it?",
#     "2) Describe a situation where you had to work with a difficult team member. How did you handle it?",
#     "3) Tell me about a time when you made a mistake at work. How did you address it?",
#     "4) How do you handle situations where you need to learn something new?",
#     "5) Can you share an example of when you had to adapt to a significant change at work?"
# ]

# MODEL_SIZE = "base"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8" if DEVICE == "cuda" else "int8")

# # -------------------- GLOBAL STATE --------------------
# session = SessionState()
# analyzer: InterviewAnalyzer | None = None

# # HR question bank (audio + text)
# # Each item: {"text": str, "audio": (sr:int, arr:np.ndarray)}
# HR_PUBLISHED = False

# # -------------------- AUDIO HELPERS --------------------
# def preprocess_audio(audio_data):
#     if audio_data is None:
#         return np.zeros((0,), dtype=np.float32), 16000
#     sr, arr = audio_data
#     if arr is None:
#         return np.zeros((0,), dtype=np.float32), 16000
#     if arr.ndim > 1 and arr.shape[1] == 2:
#         arr = arr.mean(axis=1)
#     if arr.dtype != np.float32:
#         if np.issubdtype(arr.dtype, np.integer):
#             arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
#         else:
#             arr = arr.astype(np.float32)
#     if sr != 16000:
#         arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
#         sr = 16000
#     return arr, sr

# async def transcribe(audio_data):
#     audio_arr, _ = preprocess_audio(audio_data)
#     try:
#         segments, _ = whisper_model.transcribe(
#             audio_arr, language="en", beam_size=5, vad_filter=True
#         )
#         return " ".join(seg.text for seg in segments).strip()
#     except Exception as e:
#         print(f"[transcribe] error: {e}")
#         return ""

# # -------------------- ANALYSIS SIDE --------------------
# async def run_cumulative_agent_analysis(answers_list):
#     if not answers_list:
#         return [], 0.0
#     transcript = "\n".join(answers_list)
#     try:
#         scores = await run_analysis_pipeline(transcript)
#     except Exception as e:
#         print(f"[run_cumulative_agent_analysis] run_analysis_pipeline failed: {e}")
#         scores = []
#     per_agent = []
#     total = 0.0
#     count = 0
#     for item in scores:
#         agent_name = item.get("agent_name", "Unknown")
#         score_val = float(item.get("score", 0) or 0)
#         per_agent.append({"agent_name": agent_name, "score": score_val, "evidence": item.get("evidence","")})
#         total += score_val
#         count += 1
#     overall_avg = (total / count) if count > 0 else 0.0
#     return per_agent, overall_avg

# # -------------------- UI FORMATTERS --------------------
# EIGHT_HUMILITY_AGENTS = {
#     "admitmistake", "mindchange", "learnermindset",
#     "bragflag", "blameshift", "knowitall",
#     "feedbackacceptance", "supportgrowth"
# }

# def _base_name(name: str) -> str:
#     n = (name or "").lower()
#     return n[:-5] if n.endswith("agent") else n

# def _clean_agent_name(name: str) -> str:
#     if not name: return ""
#     return name[:-5] if name.endswith("Agent") else name

# def _hide_in_live_table(name: str) -> bool:
#     n = (name or "").lower()
#     return ("pronoun" in n) or ("idontknow" in n)

# def _final_humility_from_llm(llm_agents: list[dict]) -> float:
#     vals = []
#     for a in llm_agents or []:
#         base = _base_name(a.get("agent_name",""))
#         if base in EIGHT_HUMILITY_AGENTS:
#             try: vals.append(float(a.get("score", 0.0)))
#             except Exception: pass
#     return round(sum(vals) / len(vals), 1) if vals else 0.0

# def _format_live_md(per_agent, _overall_unused):
#     shown = [a for a in per_agent if not _hide_in_live_table(a.get("agent_name",""))]
#     live_avg = (sum(float(a.get("score",0.0)) for a in shown) / len(shown)) if shown else 0.0
#     md = f"### Live Progress\n\n**LLM Agents Avg (filtered):** **{live_avg:.2f} / 10**\n\n"
#     md += "| Factor | Score |\n|---|:---:|\n"
#     for a in shown:
#         md += f"| {_clean_agent_name(a.get('agent_name',''))} | {float(a.get('score',0.0)):.2f} |\n"
#     return md

# def _format_final_table_md(llm_agents, analyzer_payload):
#     md = "## Final Agentic Analysis Summary\n\n"
#     md += "### LLM Agents\n\n"
#     md += "| Factor | Score | Evidence |\n|---|:---:|---|\n"
#     for a in llm_agents:
#         name = str(a.get("agent_name",""))
#         if any(k in name.lower() for k in ["pronoun","idontknow","praisehandling","precisehandling"]):
#             continue
#         score = float(a.get("score", 0.0))
#         ev = (a.get("evidence") or "").replace("\n", " ").strip()
#         md += f"| {_clean_agent_name(name)} | {score:.2f} | {ev[:180]}{'…' if len(ev)>180 else ''} |\n"
#     hum = _final_humility_from_llm(llm_agents)
#     md += "\n\n### Core Behavioral Traits (Summary)\n\n| Trait | Score |\n|---|:---:|\n"
#     md += f"| Humility | {hum:.1f} |\n"
#     return md

# # -------------------- HR STUDIO LOGIC --------------------
# def _hr_list_markdown(hr_list: list[dict]) -> str:
#     if not hr_list:
#         return "_No HR questions added yet._"
#     md = "### HR Question Bank\n\n| # | Question (transcribed) |\n|---:|---|\n"
#     for i, q in enumerate(hr_list, 1):
#         md += f"| {i} | {q['text']} |\n"
#     return md

# async def hr_add_question(audio, hr_list):
#     if audio is None:
#         return hr_list, _hr_list_markdown(hr_list), None, "No audio recorded."
#     text = await transcribe(audio)
#     if not text:
#         return hr_list, _hr_list_markdown(hr_list), None, "Could not transcribe. Try again."
#     # store original audio (sr, arr)
#     sr, arr = audio
#     new = {"text": text, "audio": (sr, arr)}
#     hr_list = (hr_list or []) + [new]
#     return hr_list, _hr_list_markdown(hr_list), audio, f"Added Q{len(hr_list)}."

# def hr_clear_questions():
#     return [], _hr_list_markdown([]), None, "Cleared HR questions."

# def hr_publish(toggle, hr_list):
#     note = "Published to candidate." if (toggle and hr_list) else "Not published."
#     return toggle, note

# # -------------------- INTERVIEW FLOW --------------------
# def _active_question_set(hr_list, hr_published):
#     if hr_published and hr_list:
#         return [q["text"] for q in hr_list], [q["audio"] for q in hr_list]
#     return DEFAULT_QUESTIONS, [None for _ in DEFAULT_QUESTIONS]

# def start_interview(name: str, hr_list, hr_published):
#     """Initialize session and analyzer; choose HR questions if published, else default 5."""
#     session.reset()
#     session.candidate_name = (name or "").strip() or "Candidate"
#     session.turns = []
#     session.start_time = datetime.now().isoformat()
#     global analyzer
#     analyzer = InterviewAnalyzer(candidate_name=session.candidate_name)

#     q_texts, q_audios = _active_question_set(hr_list, hr_published)
#     session.active_questions = q_texts
#     session.active_question_audios = q_audios

#     first_q = q_texts[0] if q_texts else "No questions available."
#     first_audio = q_audios[0] if q_audios else None
#     status_msg = f"Hello {session.candidate_name}. Starting interview."
#     # return status_msg, first_q, first_audio, gr.update(visible=True), gr.update(visible=True), ""
#     return status_msg, first_q, first_audio, gr.update(visible=True), ""

# async def analyze_and_store_turn(question, answer_text):
#     global analyzer
#     if analyzer is None:
#         analyzer = InterviewAnalyzer(candidate_name=getattr(session, "candidate_name", "Candidate"))
#     try:
#         await analyzer.analyze_response(question, answer_text)
#     except Exception as e:
#         print(f"[analyze_and_store_turn] error: {e}")

# async def submit_answer(audio_data):
#     if audio_data is None:
#         return "No audio received. Please record an answer.", "", None, "", gr.update(visible=True), gr.update(visible=True)

#     transcript = await transcribe(audio_data)
#     if transcript.strip() == "":
#         return "Could not transcribe audio. Try again.", "", None, "", gr.update(visible=True), gr.update(visible=True)

#     idx = len(session.turns)
#     questions = getattr(session, "active_questions", DEFAULT_QUESTIONS)
#     q_audios = getattr(session, "active_question_audios", [None]*len(questions))

#     if idx < len(questions):
#         turn = Turn(question=questions[idx], audio_data=audio_data, transcript=transcript)
#         session.turns.append(turn)
#     else:
#         # Shouldn't happen; guard
#         pass

#     await analyze_and_store_turn(turn.question, turn.transcript)

#     answers_so_far = [t.transcript for t in session.turns]
#     per_agent, overall_avg = await run_cumulative_agent_analysis(answers_so_far)
#     live_md = _format_live_md(per_agent, overall_avg)

#     # cache the agents for the final PDF
#     last_llm_agents_state.value = per_agent

#     # next question
#     if len(session.turns) < len(questions):
#         next_q = questions[len(session.turns)]
#         next_audio = q_audios[len(session.turns)]
#         status = f"Recorded Q{len(session.turns)}. Moving to next question."
#         show_generate = False
#     else:
#         next_q = "All questions completed. Click 'Generate Final Report' to create PDF/HTML."
#         next_audio = None
#         status = "Interview complete."
#         show_generate = True

#     return transcript, next_q, next_audio, live_md, gr.update(visible=show_generate), gr.update(visible=show_generate)

# async def generate_final_report_and_pdf(llm_agents_cache):
#     if analyzer is None or not analyzer.analysis.turns:
#         return "No interview data available. Please run the interview first.", None, None

#     payload = analyzer.to_summary_payload()
#     html = report_generator.build_full_report_html(payload, llm_agents=llm_agents_cache or [])
#     html_path = report_generator.save_html_report(html, analyzer.candidate_name)
#     pdf_path = report_generator.create_pdf_report_from_html(html, analyzer.candidate_name)
#     return html, html_path, pdf_path


# # -------------------- UI --------------------
# # CSS_HIDE_DUPLICATE_STOP = """
# # /* Hide the streaming pause/stop controls so only one 'Stop' is visible */
# # button[aria-label="Pause streaming"], button[title="Pause streaming"] { display: none !important; }
# # button:has(span:contains("Stop")):not(.gradio_audio .controls button) { display: none !important; }
# # /* Fallback for some Gradio builds */
# # button.svelte-1ipelgc[aria-label="Pause"] { display:none !important; }
# # """
# CSS_HIDE_DUPLICATE_STOP = """
# /* Hide pause controls so only the recorder's single Stop is visible */
# button[aria-label="Pause streaming"], button[title="Pause streaming"] { display: none !important; }
# button.svelte-1ipelgc[aria-label="Pause"] { display:none !important; }  /* fallback */
# """


# with gr.Blocks(title="🎙️ Interview Suite", css=CSS_HIDE_DUPLICATE_STOP) as demo:
#     gr.Markdown("## Interview Assistant — HR Audio Questions + Candidate Analysis + PDF Report")

#     # Shared States for cross-tab communication
#     hr_questions_state = gr.State([])      # list[dict{text, audio}]
#     hr_published_state = gr.State(False)   # bool
#     # shared States (add this next to your other states)
#     last_llm_agents_state = gr.State([])  # list of {"agent_name","score","evidence"}

#     with gr.Tab("🧑‍💼 HR Studio"):
#         gr.Markdown("Record custom questions by voice. Publish to make them available to the candidate.")
#         with gr.Row():
#             with gr.Column(scale=1):
#                 hr_audio = gr.Audio(sources=["microphone"], type="numpy", label="Record HR Question (Microphone)")
#                 add_q_btn = gr.Button("Transcribe & Add Question")
#                 last_added_audio = gr.Audio(label="Last Added Question (Playback)", interactive=False)
#                 hr_status = gr.Markdown("")
#             with gr.Column(scale=1):
#                 hr_list_md = gr.Markdown(_hr_list_markdown([]))
#                 clear_btn = gr.Button("Clear HR Questions", variant="secondary")
#                 publish_toggle = gr.Checkbox(label="Publish questions to candidate", value=False)
#                 synth_toggle = gr.Checkbox(label="Synthesize TTS for last transcribed question (ElevenLabs)", value=False)
#                 synth_btn = gr.Button("Generate TTS for Last Question")
#                 synthesized_audio = gr.Audio(label="Synthesized HR TTS (Playback)", interactive=False)
#                 publish_note = gr.Markdown("_Not published._")

#         add_q_btn.click(
#             fn=hr_add_question,
#             inputs=[hr_audio, hr_questions_state],
#             outputs=[hr_questions_state, hr_list_md, last_added_audio, hr_status]
#         )
#         clear_btn.click(
#             fn=hr_clear_questions,
#             inputs=None,
#             outputs=[hr_questions_state, hr_list_md, last_added_audio, hr_status]
#         )
#         publish_toggle.change(
#             fn=hr_publish,
#             inputs=[publish_toggle, hr_questions_state],
#             outputs=[hr_published_state, publish_note]
#         )

#     with gr.Tab("🎧 Candidate Interview"):
#         with gr.Row():
#             with gr.Column(scale=1):
#                 name_input = gr.Textbox(label="Candidate Name", placeholder="Enter candidate name")
#                 start_btn = gr.Button("Start Interview")

#                 question_display = gr.Textbox(label="Current Question", interactive=False, lines=3)
#                 question_audio_player = gr.Audio(label="HR Question Audio (Playback)", type="numpy", interactive=False)

#                 audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record your answer", visible=False)
#                 #submit_btn = gr.Button("Submit Answer", visible=False)
#                 generate_report_btn = gr.Button("Generate Final Report (HTML + PDF)", visible=False)

#                 status = gr.Markdown("")
#             with gr.Column(scale=1):
#                 transcript_box = gr.Textbox(label="Last Transcript", lines=4, interactive=False)
#                 live_progress = gr.Markdown("### Live progress will appear here")
#                 final_report_html = gr.HTML("")
#                 download_html = gr.File(label="Download HTML report")
#                 download_pdf = gr.File(label="Download PDF report")

#         # Start interview
#         start_btn.click(
#             fn=start_interview,
#             inputs=[name_input, hr_questions_state, hr_published_state],
#             outputs=[status, question_display, question_audio_player, audio_input ,live_progress]
#         )

#         # Submit answer
#         # submit_btn.click(
#         #     fn=submit_answer,
#         #     inputs=[audio_input],
#         #     outputs=[transcript_box, question_display, question_audio_player, live_progress, generate_report_btn, generate_report_btn]
#         # )
#         audio_input.change(
#             fn=submit_answer,
#             inputs=[audio_input],
#             outputs=[transcript_box, question_display, question_audio_player, live_progress, generate_report_btn, generate_report_btn],
#         )

#         def hr_synthesize_last(hr_list, synth_toggle):
#             if not (hr_list and synth_toggle):
#                 return None, "TTS disabled or no questions."
#             last = hr_list[-1]
#             sr_arr = synth_tts_eleven(last["text"])
#             if not sr_arr:
#                 return None, "TTS unavailable (check ELEVEN_API_KEY or package)."
#             sr, arr = sr_arr
#             # replace (or attach) synthesized audio for last question
#             last["audio"] = (sr, arr)
#             return (sr, arr), "TTS generated and attached to last question."

#             synth_btn.click(
#             fn=hr_synthesize_last,
#             inputs=[hr_questions_state, synth_toggle],
#             outputs=[synthesized_audio, hr_status]
#         )

#         # Generate final report
#         # If you want to include the final per-agent table, capture it after the last submit; for simplicity pass empty here.
#         generate_report_btn.click(
#             fn=generate_final_report_and_pdf,
#             inputs=[last_llm_agents_state],   # <-- pass cached agents
#             outputs=[final_report_html, download_html, download_pdf]
#         )






# if __name__ == "__main__":
#     demo.launch(server_name="localhost", server_port=7860)





















# # interview_gradio_app.py
# import os
# import io
# import numpy as np
# import torch
# import gradio as gr
# import librosa
# import soundfile as sf
# from faster_whisper import WhisperModel
# from datetime import datetime

# from interview_state import SessionState, Turn
# from interview_analyzer import InterviewAnalyzer
# from backend.agent_manager import run_analysis_pipeline
# import report_generator  # unchanged
# from dotenv import load_dotenv
# load_dotenv()

# # -------------------- ElevenLabs TTS (ONLY) --------------------
# def synth_tts_eleven(text: str, voice_id: str | None = None) -> tuple[int, np.ndarray]:
#     """
#     Return (sr=16000, float32 mono) synthesized with ElevenLabs ONLY (new SDK).
#     Requires: pip install elevenlabs  and ELEVEN_API_KEY env var.
#     """
#     try:
#         from elevenlabs import VoiceSettings
#         from elevenlabs.client import ElevenLabs
#     except Exception as e:
#         raise RuntimeError("ElevenLabs SDK not installed. Run: pip install elevenlabs") from e

#     api_key = os.getenv("ELEVEN_API_KEY", "")
#     if not api_key:
#         raise RuntimeError("ELEVEN_API_KEY env var not set.")

#     client = ElevenLabs(api_key=api_key)
#     v_id = voice_id or os.getenv("ELEVEN_VOICE_ID", "Rachel")
#     model_id = os.getenv("ELEVEN_MODEL_ID", "eleven_multilingual_v2")

#     try:
#         # NOTE: use convert(...) (generator of bytes chunks), not convert_as_stream(...)
#         chunk_iter = client.text_to_speech.convert(
#             voice_id=v_id,
#             model_id=model_id,
#             text=text,
#             output_format="pcm_16000",
#             optimize_streaming_latency="0",
#             voice_settings=VoiceSettings(
#                 stability=0.35,
#                 similarity_boost=0.85,
#                 style=0.3,
#                 use_speaker_boost=True,
#             ),
#         )

#         # Concatenate streamed chunks
#         pcm = b"".join(chunk for chunk in chunk_iter)
#         if not pcm:
#             raise RuntimeError("ElevenLabs returned empty audio stream.")
#         arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
#         return 16000, arr

#     except Exception as e:
#         raise RuntimeError(f"ElevenLabs TTS failed: {e}") from e


# # -------------------- CONFIG --------------------
# os.makedirs("reports", exist_ok=True)
# DEFAULT_QUESTIONS = [
#     "1) Can you tell me about a time when you received constructive criticism? How did you handle it?",
#     "2) Describe a situation where you had to work with a difficult team member. How did you handle it?",
#     "3) Tell me about a time when you made a mistake at work. How did you address it?",
#     "4) How do you handle situations where you need to learn something new?",
#     "5) Can you share an example of when you had to adapt to a significant change at work?"
# ]

# MODEL_SIZE = "base"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# whisper_model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type="int8" if DEVICE == "cuda" else "int8")

# # -------------------- GLOBAL STATE --------------------
# session = SessionState()
# analyzer: InterviewAnalyzer | None = None

# # -------------------- AUDIO HELPERS --------------------
# def preprocess_audio(audio_data):
#     if audio_data is None:
#         return np.zeros((0,), dtype=np.float32), 16000
#     sr, arr = audio_data
#     if arr is None:
#         return np.zeros((0,), dtype=np.float32), 16000
#     if arr.ndim > 1 and arr.shape[1] == 2:
#         arr = arr.mean(axis=1)
#     if arr.dtype != np.float32:
#         if np.issubdtype(arr.dtype, np.integer):
#             arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
#         else:
#             arr = arr.astype(np.float32)
#     if sr != 16000:
#         arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
#         sr = 16000
#     return arr, sr

# async def transcribe(audio_data):
#     audio_arr, _ = preprocess_audio(audio_data)
#     try:
#         segments, _ = whisper_model.transcribe(
#             audio_arr, language="en", beam_size=5, vad_filter=True
#         )
#         return " ".join(seg.text for seg in segments).strip()
#     except Exception as e:
#         print(f"[transcribe] error: {e}")
#         return ""

# # -------------------- ANALYSIS --------------------
# async def run_cumulative_agent_analysis(answers_list):
#     if not answers_list:
#         return [], 0.0
#     transcript = "\n".join(answers_list)
#     try:
#         scores = await run_analysis_pipeline(transcript)
#     except Exception as e:
#         print(f"[run_cumulative_agent_analysis] failed: {e}")
#         scores = []
#     per_agent, total, count = [], 0.0, 0
#     for item in scores:
#         agent_name = item.get("agent_name", "Unknown")
#         score_val = float(item.get("score", 0) or 0)
#         per_agent.append({"agent_name": agent_name, "score": score_val, "evidence": item.get("evidence","")})
#         total += score_val
#         count += 1
#     overall_avg = (total / count) if count > 0 else 0.0
#     return per_agent, overall_avg

# # -------------------- UI FORMATTERS --------------------
# EIGHT_HUMILITY_AGENTS = {
#     "admitmistake", "mindchange", "learnermindset",
#     "bragflag", "blameshift", "knowitall",
#     "feedbackacceptance", "supportgrowth"
# }

# def _base_name(name: str) -> str:
#     n = (name or "").lower()
#     return n[:-5] if n.endswith("agent") else n

# def _clean_agent_name(name: str) -> str:
#     if not name: return ""
#     return name[:-5] if name.endswith("Agent") else name

# def _hide_in_live_table(name: str) -> bool:
#     n = (name or "").lower()
#     return ("pronoun" in n) or ("idontknow" in n)

# def _final_humility_from_llm(llm_agents: list[dict]) -> float:
#     vals = []
#     for a in llm_agents or []:
#         base = _base_name(a.get("agent_name",""))
#         if base in EIGHT_HUMILITY_AGENTS:
#             try:
#                 vals.append(float(a.get("score", 0.0)))
#             except Exception:
#                 pass
#     return round(sum(vals) / len(vals), 1) if vals else 0.0

# def _format_live_md(per_agent, _overall_unused):
#     shown = [a for a in per_agent if not _hide_in_live_table(a.get("agent_name",""))]
#     live_avg = (sum(float(a.get("score",0.0)) for a in shown) / len(shown)) if shown else 0.0
#     md = f"### Live Progress\n\n**LLM Factors Avg (filtered):** **{live_avg:.2f} / 10**\n\n"
#     md += "| Factor | Score |\n|---|:---:|\n"
#     for a in shown:
#         md += f"| {_clean_agent_name(a.get('agent_name',''))} | {float(a.get('score',0.0)):.2f} |\n"
#     return md

# def _format_final_table_md(llm_agents, analyzer_payload):
#     md = "## Final Agentic Analysis Summary\n\n"
#     md += "### LLM Factors\n\n"
#     md += "| Factor | Score | Evidence |\n|---|:---:|---|\n"
#     for a in llm_agents:
#         name = str(a.get("agent_name",""))
#         if any(k in name.lower() for k in ["pronoun","idontknow","praisehandling","precisehandling"]):
#             continue
#         score = float(a.get("score", 0.0))
#         ev = (a.get("evidence") or "").replace("\n", " ").strip()
#         md += f"| {_clean_agent_name(name)} | {score:.2f} | {ev[:180]}{'…' if len(ev)>180 else ''} |\n"
#     hum = _final_humility_from_llm(llm_agents)
#     md += "\n\n### Core Behavioral Traits (Summary)\n\n| Trait | Score |\n|---|:---:|\n"
#     md += f"| Humility | {hum:.1f} |\n"
#     return md

# # -------------------- HR STUDIO HELPERS --------------------
# def _hr_list_markdown(hr_list: list[dict]) -> str:
#     if not hr_list:
#         return "_No HR questions added yet._"
#     md = "### HR Question Bank\n\n| # | Question (transcribed) |\n|---:|---|\n"
#     for i, q in enumerate(hr_list, 1):
#         md += f"| {i} | {q['text']} |\n"
#     return md

# async def hr_add_question(audio, hr_list):
#     if audio is None:
#         return hr_list, _hr_list_markdown(hr_list), None, "No audio recorded."
#     text = await transcribe(audio)
#     if not text:
#         return hr_list, _hr_list_markdown(hr_list), None, "Could not transcribe. Try again."
#     sr, arr = audio
#     new = {"text": text, "audio": (sr, arr)}  # store original HR audio (optional)
#     hr_list = (hr_list or []) + [new]
#     return hr_list, _hr_list_markdown(hr_list), audio, f"Added Q{len(hr_list)}."

# def hr_clear_questions():
#     return [], _hr_list_markdown([]), None, "Cleared HR questions."

# def hr_publish(toggle, hr_list):
#     note = "Published to candidate." if (toggle and hr_list) else "Not published."
#     return toggle, note

# def hr_synthesize_last(hr_list, synth_toggle, voice_id):
#     if not (hr_list and synth_toggle):
#         return None, "TTS disabled or no questions."
#     last = hr_list[-1]
#     try:
#         out = synth_tts_eleven(last["text"], voice_id=voice_id or None)
#         last["audio"] = out
#         return out, "TTS (ElevenLabs) generated and attached to last question."
#     except Exception as e:
#         return None, f"TTS error: {e}"

# # -------------------- INTERVIEW FLOW --------------------
# def _active_question_set(hr_list, hr_published, voice_id) -> tuple[list[str], list[tuple[int, np.ndarray] | None], str]:
#     """
#     Returns (texts, audios, warning)
#     - If HR published: use HR list (audio may be HR’s or ElevenLabs synthesized per-button).
#       If any HR question lacks audio, synthesize it now with ElevenLabs.
#     - Else: default 5 synthesized with ElevenLabs.
#     """
#     warning = ""
#     if hr_published and hr_list:
#         texts = [q["text"] for q in hr_list]
#         audios = []
#         for q in hr_list:
#             if q.get("audio") is not None:
#                 audios.append(q["audio"])
#             else:
#                 try:
#                     audios.append(synth_tts_eleven(q["text"], voice_id=voice_id or None))
#                 except Exception as e:
#                     audios.append(None)
#                     warning = f"⚠️ TTS error for an HR question: {e}"
#         return texts, audios, warning

#     # Default 5 — synth ALL with ElevenLabs
#     texts = DEFAULT_QUESTIONS[:]
#     audios = []
#     try:
#         for q in texts:
#             audios.append(synth_tts_eleven(q, voice_id=voice_id or None))
#     except Exception as e:
#         # If any failure occurs, stop and show warning; candidate will see the text but no audio
#         warning = f"⚠️ ElevenLabs TTS failed: {e}"
#         audios = [None] * len(texts)
#     return texts, audios, warning

# def start_interview(name: str, hr_list, hr_published, voice_id):
#     session.reset()
#     session.candidate_name = (name or "").strip() or "Candidate"
#     session.turns = []
#     session.start_time = datetime.now().isoformat()
#     global analyzer
#     analyzer = InterviewAnalyzer(candidate_name=session.candidate_name)

#     q_texts, q_audios, warn = _active_question_set(hr_list, hr_published, voice_id)
#     session.active_questions = q_texts
#     session.active_question_audios = q_audios

#     first_q = q_texts[0] if q_texts else "No questions available."
#     first_audio = q_audios[0] if q_audios else None
#     status_msg = f"Hello {session.candidate_name}. Starting interview."
#     if warn:
#         status_msg += f"\n\n{warn}"
#     return status_msg, first_q, first_audio, gr.update(visible=True), ""  # recorder visible; progress blank

# async def analyze_and_store_turn(question, answer_text):
#     global analyzer
#     if analyzer is None:
#         analyzer = InterviewAnalyzer(candidate_name=getattr(session, "candidate_name", "Candidate"))
#     try:
#         await analyzer.analyze_response(question, answer_text)
#     except Exception as e:
#         print(f"[analyze_and_store_turn] error: {e}")

# async def submit_answer(audio_data):
#     """
#     Auto-called when candidate presses Stop on the recorder.
#     """
#     if audio_data is None:
#         return "No audio received. Please record an answer.", "", None, "", gr.update(visible=True), gr.update(visible=True)

#     transcript = await transcribe(audio_data)
#     if transcript.strip() == "":
#         return "Could not transcribe audio. Try again.", "", None, "", gr.update(visible=True), gr.update(visible=True)

#     idx = len(session.turns)
#     questions = getattr(session, "active_questions", DEFAULT_QUESTIONS)
#     q_audios = getattr(session, "active_question_audios", [None]*len(questions))

#     if idx < len(questions):
#         turn = Turn(question=questions[idx], audio_data=audio_data, transcript=transcript)
#         session.turns.append(turn)
#         await analyze_and_store_turn(turn.question, turn.transcript)

#     # cumulative factors
#     answers_so_far = [t.transcript for t in session.turns]
#     per_agent, overall_avg = await run_cumulative_agent_analysis(answers_so_far)
#     live_md = _format_live_md(per_agent, overall_avg)
#     # cache for final PDF
#     last_llm_agents_state.value = per_agent

#     # next question or finish
#     if len(session.turns) < len(questions):
#         next_q = questions[len(session.turns)]
#         next_audio = q_audios[len(session.turns)]
#         status = f"Recorded Q{len(session.turns)}. Moving to next question."
#         show_generate = False
#     else:
#         next_q = "All questions completed. Click 'Generate Final Report' to create PDF/HTML."
#         next_audio = None
#         status = "Interview complete."
#         show_generate = True

#     return transcript, next_q, next_audio, live_md, gr.update(visible=show_generate), gr.update(visible=show_generate)

# async def generate_final_report_and_pdf(llm_agents_cache):
#     if analyzer is None or not analyzer.analysis.turns:
#         return "No interview data available. Please run the interview first.", None, None
#     payload = analyzer.to_summary_payload()
#     html = report_generator.build_full_report_html(payload, llm_agents=llm_agents_cache or [])
#     html_path = report_generator.save_html_report(html, analyzer.candidate_name)
#     pdf_path = report_generator.create_pdf_report_from_html(html, analyzer.candidate_name)
#     return html, html_path, pdf_path

# # -------------------- UI --------------------
# CSS_HIDE_DUPLICATE_STOP = """
# /* Hide pause controls so only the recorder's single Stop is visible */
# button[aria-label="Pause streaming"], button[title="Pause streaming"] { display: none !important; }
# button.svelte-1ipelgc[aria-label="Pause"] { display:none !important; }  /* fallback */
# """

# with gr.Blocks(title="🎙️ Interview Suite", css=CSS_HIDE_DUPLICATE_STOP) as demo:
#     gr.Markdown("## Interview Assistant — HR Audio Questions + Candidate Analysis + PDF Report")

#     # Shared states
#     hr_questions_state = gr.State([])        # list[{"text": str, "audio": (sr, arr)}]
#     hr_published_state = gr.State(False)     # bool
#     last_llm_agents_state = gr.State([])     # list of {"agent_name","score","evidence"}
#     voice_id_state = gr.State("")            # ElevenLabs voice id (optional)

#     with gr.Tab("🧑‍💼 HR Studio"):
#         gr.Markdown("Record custom questions by voice. Publish to make them available to the candidate.")
#         with gr.Row():
#             with gr.Column(scale=1):
#                 eleven_voice = gr.Textbox(label="ElevenLabs Voice ID (optional)", placeholder="e.g. Rachel (leave blank to use ELEVEN_VOICE_ID env var)")
#                 set_voice_btn = gr.Button("Use This Voice")
#                 hr_audio = gr.Audio(sources=["microphone"], type="numpy", label="Record HR Question (Microphone)")
#                 add_q_btn = gr.Button("Transcribe & Add Question")
#                 last_added_audio = gr.Audio(label="Last Added Question (Playback)", type="numpy", interactive=False)
#                 hr_status = gr.Markdown("")
#             with gr.Column(scale=1):
#                 hr_list_md = gr.Markdown(_hr_list_markdown([]))
#                 clear_btn = gr.Button("Clear HR Questions", variant="secondary")
#                 publish_toggle = gr.Checkbox(label="Publish questions to candidate", value=False)
#                 synth_toggle = gr.Checkbox(label="Synthesize TTS for last transcribed question (ElevenLabs)", value=False)
#                 synth_btn = gr.Button("Generate TTS for Last Question")
#                 synthesized_audio = gr.Audio(label="Synthesized HR TTS (Playback)", type="numpy", interactive=False)
#                 publish_note = gr.Markdown("_Not published._")

#         def _set_voice(v):
#             voice_id_state.value = (v or "").strip()
#             return f"Voice set to: {voice_id_state.value or os.getenv('ELEVEN_VOICE_ID','(env default)')}"

#         set_voice_btn.click(fn=_set_voice, inputs=[eleven_voice], outputs=[hr_status])

#         add_q_btn.click(
#             fn=hr_add_question,
#             inputs=[hr_audio, hr_questions_state],
#             outputs=[hr_questions_state, hr_list_md, last_added_audio, hr_status]
#         )
#         clear_btn.click(
#             fn=hr_clear_questions,
#             inputs=None,
#             outputs=[hr_questions_state, hr_list_md, last_added_audio, hr_status]
#         )
#         publish_toggle.change(
#             fn=hr_publish,
#             inputs=[publish_toggle, hr_questions_state],
#             outputs=[hr_published_state, publish_note]
#         )
#         synth_btn.click(
#             fn=lambda q, t, v: hr_synthesize_last(q, t, v),
#             inputs=[hr_questions_state, synth_toggle, voice_id_state],
#             outputs=[synthesized_audio, hr_status]
#         )

#     with gr.Tab("🎧 Candidate Interview"):
#         with gr.Row():
#             with gr.Column(scale=1):
#                 name_input = gr.Textbox(label="Candidate Name", placeholder="Enter candidate name")
#                 start_btn = gr.Button("Start Interview")

#                 question_display = gr.Textbox(label="Current Question", interactive=False, lines=3)
#                 question_audio_player = gr.Audio(label="Question Audio (ElevenLabs)", type="numpy", autoplay=True, interactive=False)

#                 audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record your answer", visible=False)
#                 generate_report_btn = gr.Button("Generate Final Report (HTML + PDF)", visible=False)

#                 status = gr.Markdown("")

#             with gr.Column(scale=1):
#                 transcript_box = gr.Textbox(label="Last Transcript", lines=4, interactive=False)
#                 live_progress = gr.Markdown("### Live progress will appear here")
#                 final_report_html = gr.HTML("")
#                 download_html = gr.File(label="Download HTML report")
#                 download_pdf = gr.File(label="Download PDF report")

#         # Start interview (inject selected voice id)
#         start_btn.click(
#             fn=lambda name, qlist, pub, vid: start_interview(name, qlist, pub, vid),
#             inputs=[name_input, hr_questions_state, hr_published_state, voice_id_state],
#             outputs=[status, question_display, question_audio_player, audio_input, live_progress]
#         )

#         # Auto-submit when candidate presses Stop
#         audio_input.change(
#             fn=submit_answer,
#             inputs=[audio_input],
#             outputs=[transcript_box, question_display, question_audio_player, live_progress, generate_report_btn, generate_report_btn],
#         )

#         # Final report
#         generate_report_btn.click(
#             fn=generate_final_report_and_pdf,
#             inputs=[last_llm_agents_state],
#             outputs=[final_report_html, download_html, download_pdf]
#         )

# if __name__ == "__main__":
#     demo.launch(server_name="localhost", server_port=7860)























##dopher 1  ##kaam kar raha hai 
# interview_gradio_app.py
import os, io, json, tempfile, requests
from datetime import datetime
from typing import List, Tuple, Any

import numpy as np
import torch
import gradio as gr
import soundfile as sf
import librosa
from faster_whisper import WhisperModel

from interview_state import SessionState, Turn
from interview_analyzer import InterviewAnalyzer
from backend.agent_manager import run_analysis_pipeline, results_to_llm_factor_rows
import report_generator
import time

# === PROCTORING: imports (safe if package not present) ===
try:
    from proctoring.proctor_widget import proctor_widget_html  # browser-side widget (HTML+JS)
except Exception:
    proctor_widget_html = None  # type: ignore
# === end PROCTORING imports ===

# debounce
_last_event_t = 0.0
_DEBOUNCE_SEC = 0.8  # ignore back-to-back “ghost” events inside this window



# =========================
# Toggle debug panel
# =========================
SHOW_DEBUG = True  # set to False to hide debug panel in the UI

# =========================
# ElevenLabs (robust)
# =========================
_ELEVEN_VOICE_CACHE = {"by_name": {}, "by_id": set()}

def _eleven_resolve_voice_id(client, user_value: str | None) -> str | None:
    try:
        if not _ELEVEN_VOICE_CACHE["by_name"] and not _ELEVEN_VOICE_CACHE["by_id"]:
            voices_obj = None
            for getter in (lambda: client.voices.get_all(),
                           lambda: client.voices.list(),
                           lambda: client.voices()):
                try:
                    voices_obj = getter()
                    if voices_obj:
                        break
                except Exception:
                    pass
            if voices_obj:
                iterable = voices_obj.voices if hasattr(voices_obj, "voices") else voices_obj
                for v in iterable:
                    vid = getattr(v, "voice_id", None) or getattr(v, "id", None)
                    vname = (getattr(v, "name", None) or "").strip()
                    if vid:
                        _ELEVEN_VOICE_CACHE["by_id"].add(vid)
                        if vname:
                            _ELEVEN_VOICE_CACHE["by_name"][vname.lower()] = vid
        if user_value:
            uv = user_value.strip()
            if uv in _ELEVEN_VOICE_CACHE["by_id"]:
                return uv
            lname = uv.lower()
            if lname in _ELEVEN_VOICE_CACHE["by_name"]:
                return _ELEVEN_VOICE_CACHE["by_name"][lname]
        if _ELEVEN_VOICE_CACHE["by_id"]:
            return sorted(_ELEVEN_VOICE_CACHE["by_id"])[0]
        return None
    except Exception as e:
        print(f"[TTS] resolve voice failed: {e}")
        return None

def synth_tts_eleven(text: str, voice: str | None = None, **kwargs) -> tuple[int, np.ndarray] | None:
    try:
        from elevenlabs import VoiceSettings
        from elevenlabs.client import ElevenLabs
    except Exception as e:
        print(f"[TTS] elevenlabs package not installed: {e}")
        return None
    api_key = os.getenv("ELEVEN_API_KEY", "")
    if not api_key:
        print("[TTS] ELEVEN_API_KEY not set; skipping TTS.")
        return None
    user_provided = voice or kwargs.get("voice_id") or kwargs.get("voice_value")
    env_name = os.getenv("ELEVEN_VOICE_NAME", "").strip() or None
    env_id   = os.getenv("ELEVEN_VOICE_ID", "").strip() or None
    chosen_value = user_provided or env_id or env_name
    try:
        client = ElevenLabs(api_key=api_key)
    except Exception as e:
        print(f"[TTS] Could not init ElevenLabs client: {e}")
        client = None
    voice_id = None
    if client is not None:
        voice_id = _eleven_resolve_voice_id(client, chosen_value)
    if not voice_id and chosen_value:
        voice_id = chosen_value
    if not voice_id:
        print(f"[TTS] No valid ElevenLabs voice found for '{chosen_value}'.")
        return None
    # Try SDK variants then REST
    try:
        if client is not None and hasattr(client, "text_to_speech"):
            tts = client.text_to_speech
            if hasattr(tts, "convert_as_stream"):
                stream = tts.convert_as_stream(voice_id=voice_id, optimize_streaming_latency="0",
                                               output_format="pcm_16000", text=text,
                                               voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.7))
                pcm_bytes = b"".join(chunk for chunk in stream)
                arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                return 16000, arr
            if hasattr(tts, "convert"):
                pcm_bytes = tts.convert(voice_id=voice_id, optimize_streaming_latency="0",
                                        output_format="pcm_16000", text=text,
                                        voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.7))
                if not isinstance(pcm_bytes, (bytes, bytearray)):
                    try: pcm_bytes = b"".join(pcm_bytes)
                    except Exception: pcm_bytes = bytes(pcm_bytes)
                arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                return 16000, arr
            if hasattr(client, "generate"):
                audio_bytes = client.generate(text=text, voice=voice_id, model="eleven_multilingual_v2",
                                              stream=False, output_format="pcm_16000",
                                              voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.7))
                if not isinstance(audio_bytes, (bytes, bytearray)):
                    audio_bytes = bytes(audio_bytes)
                arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                return 16000, arr
    except Exception as e:
        print(f"[TTS] SDK path failed: {e}")
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        headers = {"xi-api-key": api_key, "accept": "application/octet-stream", "Content-Type": "application/json"}
        payload = {"text": text, "output_format": "pcm_16000", "optimize_streaming_latency": "0",
                   "voice_settings": {"stability": 0.5, "similarity_boost": 0.7}}
        with requests.post(url, headers=headers, data=json.dumps(payload), stream=True, timeout=60) as r:
            if r.status_code != 200:
                try: msg = r.json()
                except Exception: msg = r.text
                print(f"[TTS] REST failed [{r.status_code}]: {msg}")
                return None
            pcm_bytes = b"".join(chunk for chunk in r.iter_content(chunk_size=8192))
            arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return 16000, arr
    except Exception as e:
        print(f"[TTS] REST exception: {e}")
    return None

# =========================
# Config / Globals
# =========================
os.makedirs("reports", exist_ok=True)
DEFAULT_QUESTIONS = [
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

# =========================
# Audio helpers (robust)
# =========================
def _load_audio_file(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    try:
        arr, sr = sf.read(path, dtype="float32", always_2d=False)
        if arr.ndim == 2: arr = arr.mean(axis=1)
        if sr != target_sr:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return arr.astype(np.float32), sr
    except Exception as e:
        print(f"[fileload] failed to read {path}: {e}")
        return np.zeros((0,), dtype=np.float32), target_sr

def preprocess_audio(audio_data: Any) -> Tuple[np.ndarray, int, str]:
    """
    Accepts:
      - (sr:int, data: np.ndarray)
      - {"sample_rate": int, "data": np.ndarray/list, "path": str}  # any of these keys may exist
      - np.ndarray (assume 16k)
      - str (filepath)
      - None
    Returns: (arr: float32 mono @16k, sr=16000, src: 'numpy'|'file'|'empty')
    """
    # case: dict with path
    if isinstance(audio_data, dict) and audio_data.get("path"):
        arr, sr = _load_audio_file(audio_data["path"], 16000)
        return arr, 16000, "file"

    # tuple/list
    if isinstance(audio_data, (tuple, list)) and len(audio_data) == 2:
        sr, arr = audio_data
        if isinstance(arr, list): arr = np.asarray(arr)
        if isinstance(arr, np.ndarray):
            if arr.ndim == 2: arr = arr.mean(axis=1)
            if arr.dtype != np.float32:
                if np.issubdtype(arr.dtype, np.integer):
                    arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max or 32767)
                else:
                    arr = arr.astype(np.float32)
            if sr != 16000 and arr.size > 0:
                try: arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
                except Exception as e: print(f"[preprocess] resample fail: {e}")
            return arr, 16000, "numpy"

    # dict with samples (no path)
    if isinstance(audio_data, dict) and ("data" in audio_data):
        sr = int(audio_data.get("sample_rate") or audio_data.get("sampling_rate") or 16000)
        arr = audio_data["data"]
        if isinstance(arr, list): arr = np.asarray(arr)
        arr = np.asarray(arr) if not isinstance(arr, np.ndarray) else arr
        if arr.ndim == 2: arr = arr.mean(axis=1)
        if arr.dtype != np.float32:
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max or 32767)
            else:
                arr = arr.astype(np.float32)
        if sr != 16000 and arr.size > 0:
            try: arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            except Exception as e: print(f"[preprocess] resample fail: {e}")
        return arr, 16000, "numpy"

    # raw ndarray
    if isinstance(audio_data, np.ndarray):
        arr = audio_data
        if arr.ndim == 2: arr = arr.mean(axis=1)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr, 16000, "numpy"

    # raw filepath
    if isinstance(audio_data, str) and os.path.exists(audio_data):
        arr, sr = _load_audio_file(audio_data, 16000)
        return arr, 16000, "file"

    return np.zeros((0,), dtype=np.float32), 16000, "empty"

def _write_temp_wav(arr: np.ndarray, sr: int = 16000) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, arr, sr, subtype="PCM_16")
    return path

async def transcribe(audio_data: Any) -> Tuple[str, str]:
    """
    Returns: (transcript, debug_note)
    """
    arr, sr, src = preprocess_audio(audio_data)
    secs = arr.size / float(sr) if arr is not None else 0.0
    debug = f"payload={type(audio_data).__name__}, src={src}, sr={sr}, secs={secs:.2f}"

    if arr is None or arr.size < 800:  # ~50ms guard
        return "", debug + " | too-short"

    # Try numpy path with VAD
    try:
        segments, _ = whisper_model.transcribe(arr, language="en", beam_size=5, vad_filter=True)
        txt = " ".join(seg.text for seg in segments).strip()
        if txt:
            return txt, debug + " | numpy+VAD"
    except Exception as e:
        print(f"[transcribe] numpy VAD error: {e}")

    # Try numpy path without VAD
    try:
        segments, _ = whisper_model.transcribe(arr, language="en", beam_size=5, vad_filter=False)
        txt = " ".join(seg.text for seg in segments).strip()
        if txt:
            return txt, debug + " | numpy no-VAD"
    except Exception as e:
        print(f"[transcribe] numpy no-VAD error: {e}")

    # Fall back to writing a temp wav and decoding
    try:
        wav = _write_temp_wav(arr, 16000)
        segments, _ = whisper_model.transcribe(wav, language="en", beam_size=5, vad_filter=False)
        txt = " ".join(seg.text for seg in segments).strip()
        os.remove(wav)
        if txt:
            return txt, debug + " | wav fallback"
    except Exception as e:
        print(f"[transcribe] wav fallback error: {e}")

    return "", debug + " | empty after all passes"

# =========================
# Cumulative live table
# =========================
def run_cumulative_agent_analysis_sync(answers_list: List[str]) -> Tuple[List[dict], float]:
    if not answers_list:
        return [], 0.0
    transcript = "\n".join(answers_list)
    resp = run_analysis_pipeline(transcript)
    per_agent_rows = results_to_llm_factor_rows(getattr(resp, "results", []) or [])
    total = sum(float(r["score"]) for r in per_agent_rows) if per_agent_rows else 0.0
    avg10 = round(total / len(per_agent_rows), 2) if per_agent_rows else 0.0
    return per_agent_rows, avg10

TEN_FACTOR_SET = {
    "AdmitMistake","MindChange","ShareCredit","LearnerMindset","BragFlag",
    "BlameShift","KnowItAll","FeedbackAcceptance","SupportGrowth","PraiseHandling",
}
def _hide_in_live_table(name: str) -> bool:
    n = (name or "").lower()
    return ("pronoun" in n) or ("idontknow" in n)
def _format_live_md(per_agent, _overall_unused):
    shown = [a for a in per_agent if not _hide_in_live_table(a.get("agent_name",""))]
    live_avg = (sum(float(a.get("score",0.0)) for a in shown) / len(shown)) if shown else 0.0
    md = f"### Live Progress\n\n**LLM Factors Avg (filtered):** **{live_avg:.2f} / 10**\n\n"
    md += "| Factor | Score |\n|---|:---:|\n"
    for a in shown:
        md += f"| {a.get('agent_name','')} | {float(a.get('score',0.0)):.2f} |\n"
    return md

# =========================
# Flow helpers
# =========================
def _hr_list_markdown(hr_list: list[dict]) -> str:
    if not hr_list:
        return "_No HR questions added yet._"
    md = "### HR Question Bank\n\n| # | Question (transcribed) |\n|---:|---|\n"
    for i, q in enumerate(hr_list, 1):
        md += f"| {i} | {q['text']} |\n"
    return md

async def hr_add_question(audio, hr_list):
    txt, dbg = await transcribe(audio)
    if not txt:
        return hr_list, _hr_list_markdown(hr_list), None, "Could not transcribe HR question."
    sr_arr = audio if isinstance(audio, tuple) else (16000, audio)
    new = {"text": txt, "audio": sr_arr}
    hr_list = (hr_list or []) + [new]
    return hr_list, _hr_list_markdown(hr_list), audio, f"Added Q{len(hr_list)}."

def hr_clear_questions():
    return [], _hr_list_markdown([]), None, "Cleared HR questions."

def hr_publish(toggle, hr_list):
    note = "Published to candidate." if (toggle and hr_list) else "Not published."
    return toggle, note

def hr_synthesize_last(hr_list, synth_toggle, voice_value):
    if not (hr_list and synth_toggle):
        return None, "TTS disabled or no questions."
    last = hr_list[-1]
    out = synth_tts_eleven(last["text"], voice=voice_value)
    if not out:
        return None, "TTS unavailable (check ELEVEN_API_KEY or voice)."
    sr, arr = out
    last["audio"] = (sr, arr)
    return (sr, arr), "TTS generated and attached to last question."

def _active_question_set(hr_list, hr_published, voice_value: str | None):
    if hr_published and hr_list:
        texts = [q["text"] for q in hr_list]
        audios = [q.get("audio") for q in hr_list]
        return texts, audios
    texts = DEFAULT_QUESTIONS[:]
    audios = [synth_tts_eleven(q, voice=voice_value) for q in texts]
    return texts, audios

def start_interview(name: str, hr_list, hr_published, voice_value: str):
    session.reset()
    session.candidate_name = (name or "").strip() or "Candidate"
    session.turns = []
    session.start_time = datetime.now().isoformat()
    global analyzer; analyzer = InterviewAnalyzer(candidate_name=session.candidate_name)
    q_texts, q_audios = _active_question_set(hr_list, hr_published, voice_value or None)
    session.active_questions = q_texts
    session.active_question_audios = q_audios
    if q_texts: session.start_turn(q_texts[0])
    first_q  = q_texts[0] if q_texts else "No questions available."
    first_a  = q_audios[0] if q_audios else None
    return f"Hello {session.candidate_name}. Starting interview.", first_q, first_a, gr.update(visible=True, value=None), "Ready. Submit your answer to see analysis."

def run_cumulative_and_format_md():
    answers_so_far = [t.transcript for t in session.turns]
    per_agent, overall_avg = run_cumulative_agent_analysis_sync(answers_so_far)
    return _format_live_md(per_agent, overall_avg), per_agent

def next_question_manual():
    questions = getattr(session, "active_questions", DEFAULT_QUESTIONS)
    q_audios = getattr(session, "active_question_audios", [None]*len(questions))
    if not questions:
        return ("No questions available. Click Start Interview.", None, "", "Click 'Start Interview' to begin.", gr.update(value=None, visible=True))
    if len(session.turns) >= len(questions) and (session.turns and session.turns[-1].transcript):
        return ("All questions completed. Click 'Generate Final Report' to create PDF/HTML.", None, "", "Interview complete.", gr.update(value=None, visible=True))
    if not session.turns or (session.turns and session.turns[-1].transcript):
        idx = len(session.turns); session.start_turn(questions[idx])
    idx = len(session.turns) - 1
    return (questions[idx], q_audios[idx], "", f"Moved to Q{idx+1} of {len(questions)}.", gr.update(value=None, visible=True))

# =========================
# Submit handlers
# =========================
async def analyze_and_store_turn(question, answer_text):
    global analyzer
    if analyzer is None:
        analyzer = InterviewAnalyzer(candidate_name=getattr(session, "candidate_name", "Candidate"))
    try:
        await analyzer.analyze_response(question, answer_text)
    except Exception as e:
        print(f"[analyze_and_store_turn] error: {e}")

def _evidence_md_from_last_turn() -> str:
    last = session.turns[-1] if session.turns else None
    if not last or not getattr(last, "per_agent_rows", None):
        return "No detailed evidence yet."
    lines = [f"### Detailed Evidence — Q: {last.question}",
             "| Factor | Score | Evidence |", "|---|:---:|---|"]
    for row in last.per_agent_rows:
        name = str(row.get("agent_name", ""))
        score = float(row.get("score", 0.0))
        ev = (row.get("evidence") or "").replace("\n", " ").strip()
        lines.append(f"| {name} | {score:.2f} | {ev} |")
    lines += ["", f"**Per-Question Indices (0–10):**",
              f"- Humility: {getattr(last, 'humility_10', 0.0):.2f}",
              f"- Learning: {getattr(last, 'learning_10', 0.0):.2f}",
              f"- Feedback: {getattr(last, 'feedback_10', 0.0):.2f}"]
    return "\n".join(lines)

async def submit_answer(audio_data):
    """
    Must return 8 outputs in order:
      transcript_box, question_display, question_audio_player, live_progress,
      audio_input (update), generate_report_btn (update), generate_report_btn (update), evidence_detail_md
    """
    global _last_event_t
    now = time.time()

    # Helper: get current UI values without changing state
    def _current_ui_snapshot():
        questions = getattr(session, "active_questions", DEFAULT_QUESTIONS)
        q_audios  = getattr(session, "active_question_audios", [None]*len(questions))
        # determine current index: if last turn has transcript, we are on next question, else current
        if not session.turns:
            cur_idx = 0
        else:
            cur_idx = len(session.turns)
            if session.turns[-1].transcript == "":
                cur_idx = len(session.turns) - 1
        cur_idx = max(0, min(cur_idx, len(questions)-1))
        cur_q   = questions[cur_idx] if questions else "No question."
        cur_aud = q_audios[cur_idx] if q_audios else None

        # live progress & evidence as of now
        live_md, _ = run_cumulative_and_format_md()
        ev_md = _evidence_md_from_last_turn()
        last_transcript = session.turns[-1].transcript if session.turns else ""

        # show generate when finished
        show_generate = len(session.turns) >= len(questions) and (not session.turns or session.turns[-1].transcript != "")

        return (
            last_transcript,      # transcript_box
            cur_q,                # question_display
            cur_aud,              # question_audio_player
            live_md,              # live_progress
            gr.update(value=None, visible=True),             # reset mic (no change)
            gr.update(visible=show_generate),                # generate button vis
            gr.update(visible=show_generate),                # duplicate target
            ev_md,                                           # evidence md
        )

    # DEBOUNCE: if the event is empty and arrived too soon after the last real one → no-op
    if (audio_data is None) and (now - _last_event_t < _DEBOUNCE_SEC):
        if SHOW_DEBUG: print("[DEBUG] debounced empty event")
        return _current_ui_snapshot()

    # Try to transcribe
    transcript, dbg = await transcribe(audio_data)
    if SHOW_DEBUG: print("[DEBUG]", dbg)

    # If still empty, treat as no-op (don’t break the flow)
    if not transcript.strip():
        return _current_ui_snapshot()

    # From here we have a valid answer → update last_event_t
    _last_event_t = now

    questions = getattr(session, "active_questions", DEFAULT_QUESTIONS)
    q_audios  = getattr(session, "active_question_audios", [None]*len(questions))

    # Ensure we have an active turn
    if not session.turns:
        session.start_turn(questions[0])

    # Fill current turn if empty, otherwise append next turn with the transcript
    if session.turns and session.turns[-1].transcript == "":
        session.turns[-1].transcript = transcript
        if not session.turns[-1].user_text:
            session.turns[-1].user_text = transcript
    else:
        next_idx = min(len(session.turns), len(questions)-1)
        session.turns.append(Turn(question=questions[next_idx], transcript=transcript, user_text=transcript))

    # Analyze this turn + update analyzer summary
    session.analyze_turn(len(session.turns)-1)
    await analyze_and_store_turn(session.turns[-1].question, session.turns[-1].transcript)

    # Update live table & evidence
    live_md, per_agent_live = run_cumulative_and_format_md()
    last_llm_agents_state.value = per_agent_live
    ev_md = _evidence_md_from_last_turn()

    # Decide next question
    if len(session.turns) < len(questions):
        next_q = questions[len(session.turns)]
        next_audio = q_audios[len(session.turns)]
        show_generate = False
    else:
        next_q = "All questions completed. Click 'Generate Final Report' to create PDF/HTML."
        next_audio = None
        show_generate = True

    return (
        transcript,                    # transcript_box
        next_q,                        # question_display
        next_audio,                    # question_audio_player
        live_md,                       # live_progress
        gr.update(value=None, visible=True),  # reset mic
        gr.update(visible=show_generate),
        gr.update(visible=show_generate),
        ev_md,                         # evidence md
    )


def submit_typed_answer(answer_text: str):
    evidence_md = "No detailed evidence yet."
    if not answer_text.strip():
        return ("Please type an answer first.", "", None, "",
                gr.update(value=None, visible=True), evidence_md)

    questions = getattr(session, "active_questions", DEFAULT_QUESTIONS)
    q_audios  = getattr(session, "active_question_audios", [None]*len(questions))

    if not getattr(session, "turns", None):
        session.start_turn(questions[0])

    if session.turns and not session.turns[-1].transcript:
        session.turns[-1].transcript = answer_text
        if not session.turns[-1].user_text: session.turns[-1].user_text = answer_text
    else:
        idx = min(len(session.turns), len(questions)-1)
        session.turns.append(Turn(question=questions[idx], transcript=answer_text, user_text=answer_text))

    session.analyze_turn(len(session.turns)-1)

    live_md, per_agent = run_cumulative_agent_analysis_sync([t.transcript for t in session.turns])
    last_llm_agents_state.value = per_agent
    evidence_md = _evidence_md_from_last_turn()

    if len(session.turns) < len(questions):
        next_q = questions[len(session.turns)]
        next_audio = q_audios[len(session.turns)]
    else:
        next_q = "All questions completed. Click 'Generate Final Report' to create PDF/HTML."
        next_audio = None

    return (answer_text, next_q, next_audio, live_md, gr.update(value=None, visible=True), evidence_md)

# =========================
# Final report (no appendix)
# =========================
async def generate_final_report_and_pdf(llm_agents_cache):
    if analyzer is None:
        return "No interview data available. Please run the interview first.", None, None
    if not analyzer.analysis.turns and getattr(session, "turns", None):
        for t in session.turns:
            if t.transcript.strip():
                await analyzer.analyze_response(t.question, t.transcript)
    if not analyzer.analysis.turns:
        return "No interview data available. Please run the interview first.", None, None
    full_transcript = "\n".join([t.answer for t in analyzer.analysis.turns])
    resp = run_analysis_pipeline(full_transcript)
    overall_rows = results_to_llm_factor_rows(getattr(resp, "results", []) or [])
    payload = analyzer.to_summary_payload()
    html = report_generator.build_full_report_html(payload, llm_agents=overall_rows)

    # === PROCTORING: try to inject the Proctoring section (non-breaking) ===
    try:
        if hasattr(report_generator, "build_full_report_html_with_proctoring"):
            html_with = report_generator.build_full_report_html_with_proctoring(
                analysis_payload=payload, llm_agents=overall_rows, state=session
            )
            if html_with:
                html = html_with
    except Exception as _e:
        # Keep original html if anything goes wrong
        pass
    # === end PROCTORING injection ===

    html_path = report_generator.save_html_report(html, analyzer.candidate_name)
    pdf_path = report_generator.create_pdf_report_from_html(html, analyzer.candidate_name)
    return html, html_path, pdf_path

# === PROCTORING: server-side cache hook ===
def _cache_proctor_stats(stats_json: str):
    """
    Called whenever the hidden textbox updates (roughly once per second while proctoring runs).
    Persists the browser-side JSON into the SessionState so the report can include it.
    """
    if hasattr(session, "set_proctor_stats"):
        try:
            session.set_proctor_stats(stats_json or "")
        except Exception as e:
            if SHOW_DEBUG: print("[PROCTOR] set_proctor_stats error:", e)
    return None
# === end PROCTORING cache hook ===

# =========================
# UI
# =========================
with gr.Blocks(title="🎙️ Interview Suite") as demo:
    gr.Markdown("## Interview Assistant — HR Audio Questions + Candidate Analysis + PDF Report")
    debug_md = gr.Markdown(visible=SHOW_DEBUG, value="(debug log prints to console; set SHOW_DEBUG=False to hide)")  # optional

    # Shared states
    hr_questions_state = gr.State([])
    hr_published_state = gr.State(False)
    eleven_voice_state = gr.State("")
    last_llm_agents_state = gr.State([])

    with gr.Tab("🧑‍💼 HR Studio"):
        gr.Markdown("Record custom questions by voice. Publish to make them available to the candidate.")
        with gr.Row():
            with gr.Column(scale=1):
                hr_audio = gr.Audio(sources=["microphone"], type="numpy", label="Record HR Question (Microphone)")
                add_q_btn = gr.Button("Transcribe & Add Question")
                last_added_audio = gr.Audio(label="Last Added Question (Playback)", type="numpy", interactive=False)
                hr_status = gr.Markdown("")
            with gr.Column(scale=1):
                hr_list_md = gr.Markdown(_hr_list_markdown([]))
                clear_btn = gr.Button("Clear HR Questions", variant="secondary")
                publish_toggle = gr.Checkbox(label="Publish questions to candidate", value=False)
                synth_toggle = gr.Checkbox(label="Synthesize TTS for last transcribed question (ElevenLabs)", value=False)
                eleven_voice = gr.Textbox(label="ElevenLabs Voice (name or ID, optional)", value="")
                synth_btn = gr.Button("Generate TTS for Last Question")
                synthesized_audio = gr.Audio(label="Synthesized HR TTS (Playback)", type="numpy", interactive=False)
                publish_note = gr.Markdown("_Not published._")

        add_q_btn.click(hr_add_question, [hr_audio, hr_questions_state],
                        [hr_questions_state, hr_list_md, last_added_audio, hr_status])
        clear_btn.click(hr_clear_questions, None,
                        [hr_questions_state, hr_list_md, last_added_audio, hr_status])
        publish_toggle.change(hr_publish, [publish_toggle, hr_questions_state],
                              [hr_published_state, publish_note])
        eleven_voice.change(lambda v: v, [eleven_voice], [eleven_voice_state])
        synth_btn.click(hr_synthesize_last, [hr_questions_state, synth_toggle, eleven_voice_state],
                        [synthesized_audio, hr_status])

    with gr.Tab("🎧 Candidate Interview"):
        with gr.Row():
            with gr.Column(scale=1):
                name_input = gr.Textbox(label="Candidate Name", placeholder="Enter candidate name")
                start_btn = gr.Button("Start Interview")

                question_display = gr.Textbox(label="Current Question", interactive=False, lines=3)
                question_audio_player = gr.Audio(label="Question Audio", type="numpy", autoplay=True, interactive=False)

                audio_input = gr.Audio(sources=["microphone"], type="numpy",
                                       label="Record your answer", visible=False, streaming=False)

                typed_answer = gr.Textbox(label="(Fallback) Type your answer here", lines=3, value="")
                submit_typed_btn = gr.Button("Submit Typed Answer")

                next_q_btn = gr.Button("Next Question ➜")
                generate_report_btn = gr.Button("Generate Final Report (HTML + PDF)", visible=False)

                status = gr.Markdown("")

            with gr.Column(scale=1):
                transcript_box = gr.Textbox(label="Last Transcript", lines=4, interactive=False)
                live_progress  = gr.Markdown("### Live progress will appear here")
                evidence_detail_md = gr.Markdown("### Detailed Evidence will appear here")
                final_report_html  = gr.HTML("")
                download_html = gr.File(label="Download HTML report")
                download_pdf  = gr.File(label="Download PDF report")

                # === PROCTORING: UI block (hidden textbox + widget) ===
                with gr.Accordion("Video Proctoring (Browser Only)", open=True):
                    proctor_stats_tb = gr.Textbox(
                        label="proctor_stats_hidden",
                        value="",
                        visible=False,
                        interactive=False,
                        elem_id="proctor_stats",   # JS pushes JSON here
                    )
                    if proctor_widget_html:
                        gr.HTML(
                            value=proctor_widget_html(stats_elem_id="proctor_stats"),
                            sanitize=False   # <-- IMPORTANT: allow <script> to execute
                        )

                # cache stats to server state whenever textbox changes
                proctor_stats_tb.change(_cache_proctor_stats, [proctor_stats_tb], [])
                # === end PROCTORING UI block ===

        start_btn.click(
            start_interview,
            [name_input, hr_questions_state, hr_published_state, eleven_voice_state],
            [status, question_display, question_audio_player, audio_input, live_progress],
        )

        # Some Gradio versions fire stop_recording; others only fire change.
        # Wire BOTH to the same handler with the same 8 outputs.
        audio_input.stop_recording(
            submit_answer, [audio_input],
            [transcript_box, question_display, question_audio_player, live_progress,
             audio_input, generate_report_btn, generate_report_btn, evidence_detail_md]
        )


        next_q_btn.click(
            next_question_manual, [],
            [question_display, question_audio_player, transcript_box, status, audio_input],
        )

        submit_typed_btn.click(
            submit_typed_answer, [typed_answer],
            [transcript_box, question_display, question_audio_player, live_progress, audio_input, evidence_detail_md],
        )

        generate_report_btn.click(
            generate_final_report_and_pdf, [last_llm_agents_state],
            [final_report_html, download_html, download_pdf],
        )

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "localhost")
    port_env = os.getenv("GRADIO_SERVER_PORT", "")
    server_port = int(port_env) if port_env.isdigit() else None
    demo.launch(server_name="localhost", server_port=7861, show_error=True)
