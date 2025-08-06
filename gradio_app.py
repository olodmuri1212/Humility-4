# import base64, json, requests, gradio as gr

# BACKEND = "http://127.0.0.1:8000"

# def synthesize(question: str):
#     if not question:
#         return None, "❌ Please type a question first."
#     r = requests.post(f"{BACKEND}/generate_audio", json={"text": question}, timeout=30)
#     if r.ok:
#         wav_bytes = base64.b64decode(r.json()["audio_b64"])
#         return (wav_bytes, "audio/wav"), ""
#     return None, f"TTS error: {r.status_code}"

# def analyse(question: str, transcript: str):
#     if not transcript.strip():
#         return "⚠️ No transcript received. Click the mic and answer first."
#     payload = {"question": question, "transcript": transcript}
#     r = requests.post(f"{BACKEND}/analyze", json=payload, timeout=60)
#     if not r.ok:
#         return f"❌ Analysis failed: {r.text}"
#     report = r.json()["scores"]
#     pretty = "\n".join(
#         f"**{k}** → {v['score']}\n> {v['evidence']}" for k, v in report.items()
#     )
#     return pretty

# with gr.Blocks(title="Humility Interview (Gradio)") as demo:
#     gr.Markdown("### 🤖 Interviewer\n"
#                 "1. Enter a question, click **Speak** to hear it.\n"
#                 "2. Click **🎙️ Start Mic**, answer; transcript appears live.\n"
#                 "3. Click **Analyse** to score humility.")
    
#     with gr.Row():
#         question = gr.Textbox(label="Question", value="Describe a situation where you changed your mind.")
#         speak_btn = gr.Button("🔊 Speak")
#     audio_out = gr.Audio(label="TTS audio", interactive=False)
#     tts_status = gr.Markdown()

#     speak_btn.click(fn=synthesize,
#                     inputs=question,
#                     outputs=[audio_out, tts_status])

#     # --- Live STT via Web-Speech-API ---
#     gr.Markdown("### 🎤 Your Answer (live speech-to-text in browser)")
#     transcript = gr.Textbox(label="Live transcript")

#     # hidden HTML loads the JS recognizer
#     stt_html = gr.HTML("""
#     <script>
#     (() => {
#       const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
#       if (!SpeechRecognition) {
#         document.currentScript.parentElement.innerHTML =
#           "<b style='color:red'>Web Speech API not supported in this browser.</b>";
#         return;
#       }
#       const recog = new SpeechRecognition();
#       recog.continuous = true;
#       recog.interimResults = true;
#       let textbox;
#       const findBox = () => document.querySelector("textarea[aria-label='Live transcript']");
#       // create buttons
#       const startBtn = document.createElement("button");
#       startBtn.innerText = "🎙️ Start Mic";
#       startBtn.style = "margin-right:6px";
#       const stopBtn = document.createElement("button");
#       stopBtn.innerText = "⏹️ Stop";
#       startBtn.onclick = () => (recog.start(), startBtn.disabled=true, stopBtn.disabled=false);
#       stopBtn.onclick  = () => (recog.stop(),  startBtn.disabled=false, stopBtn.disabled=true);
#       stopBtn.disabled = true;
#       document.currentScript.parentElement.append(startBtn, stopBtn);
#       recog.onresult = (evt) => {
#         textbox = textbox || findBox();
#         let final = "", interim = "";
#         for (let i=evt.resultIndex; i<evt.results.length; i++){
#           const t = evt.results[i][0].transcript;
#           evt.results[i].isFinal ? final += t + " " : interim += t;
#         }
#         textbox.value = final + interim;
#         textbox.dispatchEvent(new Event("input", {bubbles:true}));
#       };
#     })();
#     </script>
#     """)

#     analyse_btn = gr.Button("🔎 Analyse")
#     report_md   = gr.Markdown()

#     analyse_btn.click(fn=analyse,
#                       inputs=[question, transcript],
#                       outputs=report_md)

# if __name__ == "__main__":
#     # demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)
#     demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

# import base64
# import json
# import requests
# import gradio as gr
# from typing import Tuple, Optional

# BACKEND = "http://127.0.0.1:8000"

# def synthesize(question: str) -> Tuple[Optional[tuple], str]:
#     """Synthesize speech from text using the backend TTS service."""
#     if not question.strip():
#         return None, "❌ Please enter a question first."
    
#     try:
#         response = requests.post(
#             f"{BACKEND}/generate_audio",
#             json={"text": question},
#             timeout=30
#         )
#         response.raise_for_status()
#         wav_bytes = base64.b64decode(response.json()["audio_b64"])
#         return (wav_bytes, "audio/wav"), ""
#     except requests.exceptions.RequestException as e:
#         return None, f"❌ Error: Could not connect to TTS service. {str(e)}"
#     except (KeyError, json.JSONDecodeError) as e:
#         return None, f"❌ Error processing TTS response. {str(e)}"

# def analyze_response(question: str, transcript: str) -> str:
#     """Analyze the transcript for humility indicators."""
#     if not transcript.strip():
#         return "⚠️ No transcript to analyze. Please speak your answer first."
    
#     try:
#         response = requests.post(
#             f"{BACKEND}/analyze",
#             json={"question": question, "transcript": transcript},
#             timeout=60
#         )
#         response.raise_for_status()
        
#         report = response.json().get("scores", {})
#         if not report:
#             return "⚠️ No analysis results returned."
            
#         # Format the analysis results
#         results = []
#         for aspect, details in report.items():
#             score = details.get('score', 0)
#             evidence = details.get('evidence', 'No evidence found.')
#             results.append(
#                 f"### {aspect}\n"
#                 f"**Score:** {score:.2f}/1.00\n"
#                 f"**Evidence:** {evidence}\n"
#             )
            
#         return "\n\n".join(results)
        
#     except requests.exceptions.RequestException as e:
#         return f"❌ Error: Could not connect to analysis service. {str(e)}"
#     except (KeyError, json.JSONDecodeError) as e:
#         return f"❌ Error processing analysis response. {str(e)}"

# # Create the Gradio interface
# with gr.Blocks(
#     title="Humility Interview Assistant",
#     theme=gr.themes.Soft()
# ) as demo:
#     gr.Markdown("""
#     # 🤖 Humility Interview Assistant
    
#     ### How to use:
#     1. Type your interview question in the box below
#     2. Click **🔊 Speak** to hear the question
#     3. Click **🎙️ Start Mic** and speak your answer
#     4. Click **🔎 Analyze** to get feedback on your response
#     """)
    
#     with gr.Row():
#         with gr.Column(scale=2):
#             question_input = gr.Textbox(
#                 label="Interview Question",
#                 value="Tell me about a time you made a mistake and what you learned from it.",
#                 lines=3
#             )
#             speak_btn = gr.Button("🔊 Speak", variant="primary")
#             audio_output = gr.Audio(label="Question Audio", interactive=False)
#             tts_status = gr.Markdown()
            
#             speak_btn.click(
#                 fn=synthesize,
#                 inputs=question_input,
#                 outputs=[audio_output, tts_status]
#             )
            
#         with gr.Column(scale=3):
#             gr.Markdown("### 🎤 Your Response")
#             transcript = gr.Textbox(
#                 label="Live Transcript",
#                 placeholder="Your spoken response will appear here...",
#                 lines=5
#             )
            
#             # Web Speech API integration
#             stt_html = gr.HTML("""
#             <script>
#             (() => {
#                 const startBtn = document.createElement("button");
#                 startBtn.innerText = "🎙️ Start Mic";
#                 startBtn.style.marginRight = "8px";
#                 startBtn.className = "gradio-button secondary";
                
#                 const stopBtn = document.createElement("button");
#                 stopBtn.innerText = "⏹️ Stop";
#                 stopBtn.className = "gradio-button stop";
#                 stopBtn.disabled = true;
                
#                 const status = document.createElement("span");
#                 status.style.marginLeft = "8px";
#                 status.style.color = "#666";
                
#                 const container = document.createElement("div");
#                 container.style.margin = "10px 0";
#                 container.append(startBtn, stopBtn, status);
#                 document.currentScript.parentElement.prepend(container);
                
#                 let recognition = null;
                
#                 if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
#                     status.textContent = "❌ Web Speech API not supported in this browser.";
#                     return;
#                 }
                
#                 const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
#                 recognition = new SpeechRecognition();
#                 recognition.continuous = true;
#                 recognition.interimResults = true;
                
#                 startBtn.onclick = () => {
#                     try {
#                         recognition.start();
#                         startBtn.disabled = true;
#                         stopBtn.disabled = false;
#                         status.textContent = "Listening...";
#                         status.style.color = "#4CAF50";
#                     } catch (e) {
#                         status.textContent = "Error: " + e.message;
#                         status.style.color = "red";
#                     }
#                 };
                
#                 stopBtn.onclick = () => {
#                     recognition.stop();
#                     startBtn.disabled = false;
#                     stopBtn.disabled = true;
#                     status.textContent = "Ready";
#                     status.style.color = "#666";
#                 };
                
#                 recognition.onresult = (event) => {
#                     const textbox = document.querySelector("textarea[aria-label='Live Transcript']");
#                     if (!textbox) return;
                    
#                     let final = "";
#                     let interim = "";
                    
#                     for (let i = event.resultIndex; i < event.results.length; i++) {
#                         const transcript = event.results[i][0].transcript;
#                         if (event.results[i].isFinal) {
#                             final += transcript + " ";
#                         } else {
#                             interim = transcript;
#                         }
#                     }
                    
#                     textbox.value = final + interim;
#                     textbox.dispatchEvent(new Event("input", {bubbles: true}));
#                 };
                
#                 recognition.onerror = (event) => {
#                     status.textContent = "Error: " + event.error;
#                     status.style.color = "red";
#                     startBtn.disabled = false;
#                     stopBtn.disabled = true;
#                 };
                
#                 recognition.onend = () => {
#                     if (stopBtn && !stopBtn.disabled) {
#                         startBtn.disabled = false;
#                         stopBtn.disabled = true;
#                         status.textContent = "Ready";
#                         status.style.color = "#666";
#                     }
#                 };
#             })();
#             </script>
#             """)
            
#             analyze_btn = gr.Button("🔎 Analyze Response", variant="primary")
#             analysis_output = gr.Markdown(label="Analysis Results")
            
#             analyze_btn.click(
#                 fn=analyze_response,
#                 inputs=[question_input, transcript],
#                 outputs=analysis_output
#             )

# if __name__ == "__main__":
#     demo.launch(
#         server_name="0.0.0.0",
#         server_port=7860,
#         share=False,
#         show_error=True
#     )

# import base64
# import json
# import requests
# import gradio as gr
# from typing import Tuple, Optional, Dict, Any

# # Configuration
# BACKEND = "http://127.0.0.1:8000"
# SAMPLE_RATE = 24000  # Standard sample rate for TTS output

# def synthesize(question: str) -> Tuple[Optional[tuple], str]:
#     """Synthesize speech from text using the backend TTS service."""
#     if not question.strip():
#         return None, "❌ Please enter a question first."
    
#     try:
#         response = requests.post(
#             f"{BACKEND}/generate_audio",
#             json={"text": question},
#             timeout=30
#         )
#         response.raise_for_status()
        
#         # Get the base64 audio data
#         audio_data = response.json().get("audio_b64")
#         if not audio_data:
#             return None, "❌ No audio data received from the server."
            
#         # Decode the base64 data to bytes
#         try:
#             wav_bytes = base64.b64decode(audio_data)
#             # Return as a tuple with sample rate and numpy array
#             return (SAMPLE_RATE, wav_bytes), ""
#         except Exception as e:
#             return None, f"❌ Error decoding audio data: {str(e)}"
        
#     except requests.exceptions.RequestException as e:
#         return None, f"❌ Error: Could not connect to TTS service. {str(e)}"
#     except (KeyError, json.JSONDecodeError) as e:
#         return None, f"❌ Error processing TTS response. {str(e)}"

# def analyze_response(question: str, transcript: str) -> str:
#     """Analyze the transcript for humility indicators."""
#     if not transcript.strip():
#         return "❌ Please provide a transcript to analyze."
    
#     try:
#         response = requests.post(
#             f"{BACKEND}/analyze",
#             json={"question": question, "transcript": transcript},
#             timeout=60
#         )
#         response.raise_for_status()
        
#         # Format the analysis results
#         report = response.json().get("scores", {})
#         if not report:
#             return "❌ No analysis results received."
            
#         # Format the report as markdown
#         markdown = "## Analysis Results\n\n"
#         for category, details in report.items():
#             markdown += f"### {category.title()}\n"
#             markdown += f"**Score:** {details.get('score', 'N/A')}\n\n"
#             if 'evidence' in details:
#                 markdown += f"**Evidence:** {details['evidence']}\n\n"
#             if 'suggestions' in details:
#                 markdown += "**Suggestions:**\n"
#                 for suggestion in details['suggestions']:
#                     markdown += f"- {suggestion}\n"
#         return markdown
        
#     except requests.exceptions.RequestException as e:
#         return f"❌ Error: Could not connect to analysis service. {str(e)}"

# # Create the Gradio interface
# with gr.Blocks(
#     title="Humility Interview Assistant",
#     theme=gr.themes.Soft()
# ) as demo:
#     gr.Markdown("""
#     # 🤖 Humility Interview Assistant
    
#     ### How to use:
#     1. Type your interview question in the box below
#     2. Click **🔊 Speak** to hear the question
#     3. Click **🎙️ Start Mic** and speak your answer
#     4. Click **🔎 Analyze** to get feedback on your response
#     """)
    
#     with gr.Row():
#         with gr.Column(scale=2):
#             question_input = gr.Textbox(
#                 label="Interview Question",
#                 value="Tell me about a time you made a mistake and what you learned from it.",
#                 lines=3
#             )
#             speak_btn = gr.Button("🔊 Speak", variant="primary")
#             audio_output = gr.Audio(label="Question Audio", interactive=False, type="filepath")
#             tts_status = gr.Markdown()
            
#             speak_btn.click(
#                 fn=synthesize,
#                 inputs=question_input,
#                 outputs=[audio_output, tts_status]
#             )
            
#         with gr.Column(scale=3):
#             gr.Markdown("### 🎤 Your Response")
#             transcript = gr.Textbox(
#                 label="Live Transcript",
#                 placeholder="Your spoken response will appear here...",
#                 lines=5
#             )
            
#             # Audio recording component for live transcription
#             audio_recorder = gr.Audio(
#                 sources=["microphone"],
#                 type="filepath",
#                 label="Record your response",
#                 interactive=True,
#                 show_download_button=False
#             )
            
#             # JavaScript for live transcription
#             stt_html = gr.HTML("""
#             <script>
#             document.addEventListener('DOMContentLoaded', () => {
#                 const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
#                 if (!SpeechRecognition) {
#                     const error = document.createElement('div');
#                     error.innerHTML = "<p style='color: red;'>Web Speech API is not supported in this browser. Try using Chrome or Edge.</p>";
#                     document.currentScript.parentElement.appendChild(error);
#                     return;
#                 }

#                 const recognition = new SpeechRecognition();
#                 recognition.continuous = true;
#                 recognition.interimResults = true;
                
#                 // Find the audio recorder component
#                 const audioInput = document.querySelector('input[type="file"][accept="audio/*"]');
#                 if (!audioInput) {
#                     console.error('Could not find audio input');
#                     return;
#                 }
                
#                 // Create UI elements
#                 const container = document.createElement('div');
#                 container.style.margin = '10px 0';
                
#                 const startBtn = document.createElement('button');
#                 startBtn.innerText = '🎙️ Start Recording';
#                 startBtn.className = 'gradio-button secondary';
#                 startBtn.style.marginRight = '8px';
                
#                 const stopBtn = document.createElement('button');
#                 stopBtn.innerText = '⏹️ Stop';
#                 stopBtn.className = 'gradio-button stop';
#                 stopBtn.disabled = true;
                
#                 const status = document.createElement('span');
#                 status.style.marginLeft = '8px';
#                 status.style.color = '#666';
#                 status.textContent = 'Ready';
                
#                 container.append(startBtn, stopBtn, status);
#                 audioInput.parentElement.insertBefore(container, audioInput);
                
#                 // Event handlers
#                 startBtn.onclick = (e) => {
#                     e.preventDefault();
#                     recognition.start();
#                     startBtn.disabled = true;
#                     stopBtn.disabled = false;
#                     status.textContent = 'Listening...';
#                     status.style.color = '#4CAF50';
#                 };
                
#                 stopBtn.onclick = (e) => {
#                     e.preventDefault();
#                     recognition.stop();
#                     startBtn.disabled = false;
#                     stopBtn.disabled = true;
#                     status.textContent = 'Ready';
#                     status.style.color = '#666';
#                 };
                
#                 // Handle speech recognition results
#                 recognition.onresult = (event) => {
#                     let finalTranscript = '';
#                     let interimTranscript = '';
                    
#                     for (let i = event.resultIndex; i < event.results.length; i++) {
#                         const transcript = event.results[i][0].transcript;
#                         if (event.results[i].isFinal) {
#                             finalTranscript += transcript + ' ';
#                         } else {
#                             interimTranscript = transcript;
#                         }
#                     }
                    
#                     // Update the transcript textarea
#                     const textarea = document.querySelector('textarea[placeholder*="Your spoken response"]');
#                     if (textarea) {
#                         textarea.value = finalTranscript + interimTranscript;
#                         // Trigger events to update Gradio's state
#                         const inputEvent = new Event('input', { bubbles: true });
#                         const changeEvent = new Event('change', { bubbles: true });
#                         textarea.dispatchEvent(inputEvent);
#                         textarea.dispatchEvent(changeEvent);
#                     }
#                 };
                
#                 recognition.onerror = (event) => {
#                     console.error('Speech recognition error:', event.error);
#                     status.textContent = `Error: ${event.error}`;
#                     status.style.color = 'red';
#                     startBtn.disabled = false;
#                     stopBtn.disabled = true;
#                 };
                
#                 recognition.onend = () => {
#                     if (!stopBtn.disabled) {
#                         recognition.start(); // Restart recognition if it was active
#                     }
#                 };
#             });
                
#                 const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
#                 recognition = new SpeechRecognition();
#                 recognition.continuous = true;
#                 recognition.interimResults = true;
                
#                 startBtn.onclick = () => {
#                     try {
#                         recognition.start();
#                         startBtn.disabled = true;
#                         stopBtn.disabled = false;
#                         status.textContent = "Listening...";
#                         status.style.color = "#4CAF50";
#                     } catch (e) {
#                         status.textContent = "Error: " + e.message;
#                         status.style.color = "red";
#                     }
#                 };
                
#                 stopBtn.onclick = () => {
#                     recognition.stop();
#                     startBtn.disabled = false;
#                     stopBtn.disabled = true;
#                     status.textContent = "Ready";
#                     status.style.color = "#666";
#                 };
                
#                 recognition.onresult = (event) => {
#                     const textbox = document.querySelector("textarea[aria-label='Live Transcript']");
#                     if (!textbox) return;
                    
#                     let final = "";
#                     let interim = "";
                    
#                     for (let i = event.resultIndex; i < event.results.length; i++) {
#                         const transcript = event.results[i][0].transcript;
#                         if (event.results[i].isFinal) {
#                             final += transcript + " ";
#                         } else {
#                             interim = transcript;
#                         }
#                     }
                    
#                     textbox.value = final + interim;
#                     textbox.dispatchEvent(new Event("input", {bubbles: true}));
#                 };
                
#                 recognition.onerror = (event) => {
#                     status.textContent = "Error: " + event.error;
#                     status.style.color = "red";
#                     startBtn.disabled = false;
#                     stopBtn.disabled = true;
#                 };
                
#                 recognition.onend = () => {
#                     if (stopBtn && !stopBtn.disabled) {
#                         startBtn.disabled = false;
#                         stopBtn.disabled = true;
#                         status.textContent = "Ready";
#                         status.style.color = "#666";
#                     }
#                 };
#             })();
#             </script>
#             """)
            
#             analyze_btn = gr.Button("🔍 Analyze Response", variant="primary")
#             analysis_output = gr.Markdown()
            
#             analyze_btn.click(
#                 fn=analyze_response,
#                 inputs=[question_input, transcript],
#                 outputs=analysis_output
#             )

# # Add a function to process audio and update transcript
# async def process_audio(audio_path: str) -> str:
#     if not audio_path:
#         return ""
    
#     try:
#         # Here you can add code to process the audio file if needed
#         # For now, we'll just return an empty string since we're using Web Speech API for transcription
#         return ""
#     except Exception as e:
#         print(f"Error processing audio: {str(e)}")
#         return ""

# if __name__ == "__main__":
#     # Set up the audio processing callback
#     audio_recorder.change(
#         fn=process_audio,
#         inputs=[audio_recorder],
#         outputs=[transcript]
#     )
    
#     # Launch the app
#     demo.launch(
#         server_name="0.0.0.0",
#         server_port=7860,
#         share=False,
#         show_error=True
#     )























# import base64, tempfile, requests, gradio as gr, os

# BACKEND = "http://127.0.0.1:8000"

# # ─────────── TTS callback ───────────
# def speak(question: str):
#     if not question.strip():
#         return None, "❌ Enter a question."
#     try:
#         r = requests.post(
#             f"{BACKEND}/generate_audio",
#             json={"text": question},
#             timeout=30
#         )
#         r.raise_for_status()
#         audio = base64.b64decode(r.json()["audio_b64"])
#         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#         tmp.write(audio); tmp.flush(); tmp.close()
#         return tmp.name, ""
#     except Exception as e:
#         return None, f"❌ {e}"

# # ───────── Analysis callback ─────────
# def analyse(question: str, transcript: str):
#     if not transcript.strip():
#         return "⚠️ Speak first."
#     try:
#         r = requests.post(
#             f"{BACKEND}/analyze",
#             json={"question": question, "transcript": transcript},
#             timeout=60
#         )
#         r.raise_for_status()
#         scores = r.json().get("scores", {})
#         return "\n\n".join(
#             f"### {k}\n**Score:** {v['score']}\n> {v['evidence']}"
#             for k, v in scores.items()
#         ) or "No results."
#     except Exception as e:
#         return f"❌ {e}"

# # ───────── Read your stt.js ──────────
# with open(os.path.join("public","stt.js"), "r") as f:
#     stt_js = f.read()

# # ─────────── Build the Gradio UI ───────
# with gr.Blocks(title="Humility Interview Assistant") as demo:
#     gr.Markdown("## 🤖 Humility Interview Assistant")

#     # Question & TTS
#     q = gr.Textbox(
#         label="Interview Question",
#         value="Describe a situation where you changed your mind."
#     )
#     btn_speak = gr.Button("🔊 Speak")
#     audio_out = gr.Audio(label="TTS Audio", interactive=False, type="filepath")
#     tts_status = gr.Markdown()

#     btn_speak.click(fn=speak, inputs=q, outputs=[audio_out, tts_status])

#     gr.Markdown("### 🎤 Your Response")
#     transcript = gr.Textbox(label="Live Transcript", lines=5)

#     # ─── Inject STT JS + attachSTT call ───
#     demo.add_component(gr.HTML(f"""
#     <script>
#     {stt_js}
#     </script>
#     <script>
#       window.addEventListener("DOMContentLoaded", () => {{
#         attachSTT("textarea[aria-label='Live Transcript']",
#                   "{BACKEND}");
#       }});
#     </script>
#     """))

#     # Analyse button & report
#     btn_analyse = gr.Button("🔎 Analyse")
#     report = gr.Markdown()

#     btn_analyse.click(fn=analyse, inputs=[q, transcript], outputs=report)

# # ─────────── Launch ───────────
# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=7860)


















# import base64
# import tempfile
# import requests
# import gradio as gr
# import os

# BACKEND = "http://127.0.0.1:8000"

# # ─────────── TTS callback ───────────
# def speak(question: str):
#     if not question.strip():
#         return None, "❌ Enter a question."
#     try:
#         r = requests.post(
#             f"{BACKEND}/generate_audio",
#             json={"text": question},
#             timeout=30
#         )
#         r.raise_for_status()
#         audio = base64.b64decode(r.json()["audio_b64"])
#         tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#         tmp.write(audio); tmp.flush(); tmp.close()
#         return tmp.name, ""
#     except Exception as e:
#         return None, f"❌ {e}"

# # ───────── Analysis callback ─────────
# def analyse(question: str, transcript: str):
#     if not transcript.strip():
#         return "⚠️ Speak first."
#     try:
#         r = requests.post(
#             f"{BACKEND}/analyze",
#             json={"question": question, "transcript": transcript},
#             timeout=60
#         )
#         r.raise_for_status()
#         scores = r.json().get("scores", {})
#         return "\n\n".join(
#             f"### {k}\n**Score:** {v['score']}\n> {v['evidence']}"
#             for k, v in scores.items()
#         ) or "No results."
#     except Exception as e:
#         return f"❌ {e}"

# # ───────── Read the STT JS ──────────
# with open(os.path.join("public", "stt.js"), "r") as f:
#     stt_js = f.read()

# # ─────────── Build the Gradio UI ───────
# with gr.Blocks(title="Humility Interview Assistant") as demo:
#     gr.Markdown("## 🤖 Humility Interview Assistant")

#     # Question & TTS
#     q = gr.Textbox(
#         label="Interview Question",
#         value="Describe a situation where you changed your mind."
#     )
#     btn_speak = gr.Button("🔊 Speak")
#     audio_out = gr.Audio(label="TTS Audio", interactive=False, type="filepath")
#     tts_status = gr.Markdown()

#     btn_speak.click(fn=speak, inputs=q, outputs=[audio_out, tts_status])

#     gr.Markdown("### 🎤 Your Response")
#     transcript = gr.Textbox(label="Live Transcript", lines=5)

#     # ─── Inject STT JS + attachSTT call ───
#     # First inject the stt.js script
#     stt_script = """
#     <script src="/public/stt.js"></script>
#     <script>
#       window.addEventListener("DOMContentLoaded", () => {
#         const textarea = document.querySelector("textarea[aria-label='Live Transcript']");
#         if (textarea) {
#           attachSTT(textarea, "%s");
#         } else {
#           console.error("Could not find transcript textarea");
#         }
#       });
#     </script>
#     """ % BACKEND
    
#     gr.HTML(stt_script)

#     # Analyse button & report
#     btn_analyse = gr.Button("🔎 Analyse")
#     report = gr.Markdown()

#     btn_analyse.click(fn=analyse, inputs=[q, transcript], outputs=report)

# # ─────────── Launch ───────────
# # Create a route to serve the stt.js file
# demo = gr.routes.App.create_app(demo)

# # Add a route to serve static files
# import os
# from fastapi.staticfiles import StaticFiles

# # Make sure the public directory exists
# os.makedirs("public", exist_ok=True)

# # Mount the static files
# demo.app.mount("/public", StaticFiles(directory="public"), name="public")

# if __name__ == "__main__":
#     demo.launch(
#         server_name="0.0.0.0",
#         server_port=7860,
#         show_error=True
#     )


















# import base64, tempfile, requests, gradio as gr, os

# BACKEND = "http://127.0.0.1:8000"

# # ---- TTS ----------------------------------------------------------
# def speak(q: str):
#     if not q.strip():
#         return None, "❌ Enter a question."
#     r = requests.post(f"{BACKEND}/generate_audio",
#                       json={"text": q}, timeout=30)
#     r.raise_for_status()
#     wav = base64.b64decode(r.json()["audio_b64"])
#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
#     tmp.write(wav); tmp.flush(); tmp.close()
#     return tmp.name, ""

# # ---- analysis -----------------------------------------------------
# def analyse(q: str, t: str):
#     if not t.strip(): return "⚠️ Speak first."
#     r = requests.post(f"{BACKEND}/analyze",
#                       json={"question": q, "transcript": t}, timeout=60)
#     r.raise_for_status()
#     scores = r.json()["scores"]
#     return "\n\n".join(f"### {k}\n**Score:** {v['score']}\n> {v['evidence']}"
#                        for k, v in scores.items()) or "No results."

# # read JS once
# stt_js = open("public/stt.js").read()

# with gr.Blocks(title="Humility Interview") as demo:
#     gr.Markdown("## 🤖 Humility Interview Assistant")

#     q = gr.Textbox(label="Interview Question",
#                    value="Describe a situation where you changed your mind.")
#     btn_speak = gr.Button("🔊 Speak")
#     audio = gr.Audio(label="TTS", interactive=False, type="filepath")
#     status = gr.Markdown()
#     btn_speak.click(speak, q, [audio, status])

#     gr.Markdown("### 🎤 Your Response")
#     transcript = gr.Textbox(label="Live Transcript", lines=5)

#     # inject stt.js and call attachSTT AFTER DOM is ready
#     gr.HTML(f"""
#     <script>{stt_js}</script>
#     <script>
#       window.addEventListener("DOMContentLoaded", () =>
#         attachSTT("textarea[aria-label='Live Transcript']", "{BACKEND}"));
#     </script>
#     """)

#     btn_analyse = gr.Button("🔎 Analyse")
#     report = gr.Markdown()
#     btn_analyse.click(analyse, [q, transcript], report)

# if __name__ == "__main__":
#     demo.launch(server_name="0.0.0.0", server_port=7860,
#                 static_dir="public")      # makes /file=public/stt.js work












#!/usr/bin/env python3
"""
Enhanced Gradio App with Real-time Speech-to-Text Integration
Humility Interview System - Frontend
"""

import gradio as gr
import requests
import base64
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Backend Configuration
BACKEND = "http://127.0.0.1:8000"
SAMPLE_RATE = 24000

# Ensure public directory exists
PUBLIC_DIR = Path("public")
PUBLIC_DIR.mkdir(exist_ok=True)

def synthesize_speech(question: str):
    """Convert text to speech using the backend TTS service."""
    if not question.strip():
        return None, "❌ Please enter a question first."
    
    try:
        response = requests.post(
            f"{BACKEND}/generate_audio",
            json={"text": question},
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        audio_b64 = data.get("audio_b64")
        
        if not audio_b64:
            return None, "❌ No audio data received from server."
        
        # Save as temporary file for Gradio
        audio_bytes = base64.b64decode(audio_b64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            return f.name, "✅ Question audio generated successfully!"
            
    except requests.exceptions.RequestException as e:
        return None, f"❌ Connection error: {str(e)}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def analyze_transcript(question: str, transcript: str) -> str:
    """Analyze the transcript for humility indicators."""
    if not transcript.strip():
        return "⚠️ Please provide a transcript to analyze."
    
    try:
        response = requests.post(
            f"{BACKEND}/analyze",
            json={"question": question, "transcript": transcript.strip()},
            timeout=60
        )
        response.raise_for_status()
        
        data = response.json()
        scores = data.get("scores", {})
        
        if not scores:
            return "❌ No analysis results received from server."
        
        # Format results as markdown
        markdown = "## 📊 Analysis Results\n\n"
        
        overall_score = 0
        total_agents = len(scores)
        
        for agent_name, details in scores.items():
            score = details.get('score', 0)
            evidence = details.get('evidence', 'No evidence provided')
            
            overall_score += score
            
            # Format agent name
            display_name = agent_name.replace('_', ' ').title()
            
            markdown += f"### {display_name}\n"
            markdown += f"**Score**: {score}/5 {'⭐' * int(score)}\n"
            markdown += f"**Evidence**: {evidence}\n\n"
        
        # Add overall score
        if total_agents > 0:
            avg_score = overall_score / total_agents
            markdown = f"## 🎯 Overall Humility Score: {avg_score:.1f}/5\n\n" + markdown
        
        # Add recommendations
        markdown += "\n## 💡 Recommendations\n"
        if avg_score >= 4:
            markdown += "Excellent demonstration of humility! Continue showcasing these qualities.\n"
        elif avg_score >= 3:
            markdown += "Good humility indicators. Consider emphasizing learning from failures and seeking feedback.\n"
        elif avg_score >= 2:
            markdown += "Some humility shown. Focus more on acknowledging limitations and learning opportunities.\n"
        else:
            markdown += "Consider incorporating more examples of learning from mistakes and seeking help from others.\n"
        
        return markdown
        
    except requests.exceptions.RequestException as e:
        return f"❌ Connection error: Could not reach analysis service.\n\nError: {str(e)}"
    except Exception as e:
        return f"❌ Analysis error: {str(e)}"

def export_session_data(question: str, transcript: str, analysis: str) -> str:
    """Export session data to a downloadable file."""
    if not transcript.strip():
        return "❌ No transcript available to export."
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"interview_session_{timestamp}.txt"
    
    content = f"""HUMILITY INTERVIEW SESSION
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*50}

QUESTION:
{question}

CANDIDATE RESPONSE:
{transcript}

ANALYSIS RESULTS:
{analysis}

SESSION STATISTICS:
- Word Count: {len(transcript.split())}
- Character Count: {len(transcript)}
- Estimated Speaking Time: {len(transcript.split()) / 150:.1f} minutes

{'='*50}
End of Session Report
"""
    
    # Save to public directory
    filepath = PUBLIC_DIR / filename
    filepath.write_text(content, encoding='utf-8')
    
    return f"✅ Session exported to: {filename}"

def test_backend_connection():
    """Test connection to the backend services."""
    results = []
    
    # Test health endpoint
    try:
        response = requests.get(f"{BACKEND}/", timeout=5)
        if response.status_code == 200:
            results.append("✅ Backend server is online")
        else:
            results.append(f"⚠️ Backend responded with status {response.status_code}")
    except:
        results.append("❌ Cannot connect to backend server")
    
    # Test TTS
    try:
        response = requests.post(
            f"{BACKEND}/generate_audio",
            json={"text": "Test"},
            timeout=10
        )
        if response.status_code == 200:
            results.append("✅ Text-to-Speech service working")
        else:
            results.append("❌ Text-to-Speech service error")
    except:
        results.append("❌ Text-to-Speech service unavailable")
    
    # Test Analysis
    try:
        response = requests.post(
            f"{BACKEND}/analyze",
            json={"question": "Test", "transcript": "Test response"},
            timeout=15
        )
        if response.status_code == 200:
            results.append("✅ Analysis service working")
        else:
            results.append("❌ Analysis service error")
    except:
        results.append("❌ Analysis service unavailable")
    
    return "\n".join(results)

# Create the Gradio interface
def create_interface():
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto;
    }
    
    .interview-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .section-box {
        border: 2px solid #e1e5e9;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        background: #f8f9fa;
    }
    
    .stt-info {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    #stt-controls {
        background: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 2px solid #ddd;
    }
    """
    
    # STT Integration JavaScript
    stt_js = """
    <script>
    console.log('Initializing STT integration...');
    
    // Load the STT module
    function loadSTT() {
        return new Promise((resolve, reject) => {
            if (window.RealTimeSTT) {
                resolve();
                return;
            }
            
            const script = document.createElement('script');
            script.src = '/file=public/stt.js';
            script.onload = () => {
                console.log('STT script loaded successfully');
                resolve();
            };
            script.onerror = () => {
                console.error('Failed to load STT script');
                reject(new Error('Failed to load STT script'));
            };
            document.head.appendChild(script);
        });
    }
    
    // Initialize STT when DOM is ready
    document.addEventListener('DOMContentLoaded', async function() {
        console.log('DOM loaded, initializing STT...');
        
        try {
            await loadSTT();
            
            // Wait for Gradio to fully render
            setTimeout(() => {
                // Find the transcript textarea
                const transcriptTextarea = document.querySelector('textarea[data-testid="textbox"]') ||
                                         document.querySelector('textarea[placeholder*="transcript"]') ||
                                         document.querySelector('textarea[placeholder*="response"]');
                
                if (transcriptTextarea) {
                    console.log('Found transcript textarea, attaching STT...');
                    window.sttInstance = new RealTimeSTT('textarea[data-testid="textbox"]', 'http://127.0.0.1:8000');
                    
                    // Show success message
                    const statusDiv = document.getElementById('stt-status-display');
                    if (statusDiv) {
                        statusDiv.innerHTML = '<div style="color: green; font-weight: bold;">✅ Speech-to-Text Ready!</div>';
                    }
                } else {
                    console.warn('Transcript textarea not found');
                    const statusDiv = document.getElementById('stt-status-display');
                    if (statusDiv) {
                        statusDiv.innerHTML = '<div style="color: orange;">⚠️ Transcript field not found - STT may not work</div>';
                    }
                }
            }, 2000);
            
        } catch (error) {
            console.error('STT initialization failed:', error);
            const statusDiv = document.getElementById('stt-status-display');
            if (statusDiv) {
                statusDiv.innerHTML = '<div style="color: red;">❌ Speech-to-Text initialization failed</div>';
            }
        }
    });
    
    // Global function to check STT status
    window.checkSTTStatus = function() {
        if (window.sttInstance) {
            return 'STT is active and ready';
        } else {
            return 'STT is not initialized';
        }
    };
    </script>
    """
    
    with gr.Blocks(css=custom_css, title="Humility Interview Assistant") as demo:
        # Header
        gr.HTML("""
        <div class="interview-header">
            <h1>🎤 Humility Interview Assistant</h1>
            <p>AI-Powered Interview Practice with Real-time Speech Recognition</p>
        </div>
        """)
        
        # STT Status Display
        gr.HTML('<div id="stt-status-display" style="text-align: center; margin: 10px 0;"></div>')
        
        # Add the STT JavaScript
        gr.HTML(stt_js)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Question Section
                gr.HTML('<div class="section-box">')
                gr.Markdown("### 📝 Interview Question")
                
                question_input = gr.Textbox(
                    label="Enter your interview question",
                    placeholder="e.g., Tell me about a time you made a mistake and what you learned from it.",
                    lines=3,
                    value="Tell me about a time you made a mistake and what you learned from it."
                )
                
                with gr.Row():
                    speak_btn = gr.Button("🔊 Speak Question", variant="primary")
                    test_connection_btn = gr.Button("🔧 Test Backend", variant="secondary")
                
                question_audio = gr.Audio(
                    label="Generated Question Audio",
                    type="filepath",
                    interactive=False
                )
                
                tts_status = gr.Markdown("Ready to generate speech...")
                gr.HTML('</div>')
                
                # Response Section  
                gr.HTML('<div class="section-box">')
                gr.Markdown("### 🎙️ Your Response")
                
                gr.HTML("""
                <div class="stt-info">
                    <strong>📋 Instructions:</strong>
                    <ol>
                        <li>Click "🎙️ Start Recording" below</li>
                        <li>Speak your answer clearly</li>
                        <li>Click "⏹️ Stop Recording" when done</li>
                        <li>Your speech will appear in the text box below</li>
                        <li>Click "🔍 Analyze Response" for feedback</li>
                    </ol>
                    <p><strong>Keyboard Shortcuts:</strong> Ctrl+R (Start/Stop Recording), Ctrl+K (Clear)</p>
                </div>
                """)
                
                transcript_input = gr.Textbox(
                    label="Live Transcript",
                    placeholder="Your spoken response will appear here automatically...",
                    lines=6,
                    interactive=True
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Response", variant="primary", size="lg")
                    export_btn = gr.Button("💾 Export Session", variant="secondary")
                    clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
                
                gr.HTML('</div>')
            
            with gr.Column(scale=1):
                # System Status
                gr.Markdown("### 🔧 System Status")
                connection_status = gr.Markdown("Click 'Test Backend' to check system status")
                
                # Analysis Results
                gr.Markdown("### 📊 Analysis Results")
                analysis_output = gr.Markdown("""
                **Waiting for analysis...**
                
                Your humility analysis will appear here after you:
                1. Record or type your response
                2. Click "Analyze Response"
                """)
                
                # Export Status
                export_status = gr.Markdown("")
        
        # Hidden components for session management  
        session_data = gr.State({
            "question": "",
            "transcript": "", 
            "analysis": "",
            "timestamp": ""
        })
        
        # Event handlers
        def handle_speak_question(question):
            if not question.strip():
                return None, "❌ Please enter a question first"
            
            audio_file, status = synthesize_speech(question)
            return audio_file, status
        
        def handle_analyze(question, transcript):
            if not transcript.strip():
                return "⚠️ Please record or type your response first"
            
            analysis = analyze_transcript(question, transcript)
            return analysis
        
        def handle_export(question, transcript, analysis):
            if not transcript.strip():
                return "❌ No data to export"
            
            export_result = export_session_data(question, transcript, analysis)
            return export_result
        
        def handle_clear():
            return "", "", "Session cleared. Ready for new interview question.", ""
        
        # Connect event handlers
        speak_btn.click(
            fn=handle_speak_question,
            inputs=[question_input],
            outputs=[question_audio, tts_status]
        )
        
        test_connection_btn.click(
            fn=test_backend_connection,
            outputs=[connection_status]
        )
        
        analyze_btn.click(
            fn=handle_analyze,
            inputs=[question_input, transcript_input],
            outputs=[analysis_output]
        )
        
        export_btn.click(
            fn=handle_export,
            inputs=[question_input, transcript_input, analysis_output],
            outputs=[export_status]
        )
        
        clear_btn.click(
            fn=handle_clear,
            outputs=[question_input, transcript_input, analysis_output, export_status]
        )
        
        # Auto-test connection on startup
        demo.load(
            fn=test_backend_connection,
            outputs=[connection_status]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; border-top: 1px solid #ddd;">
            <p><strong>Humility Interview System v3.2</strong></p>
            <p>🎯 Focus Areas: Learning from Mistakes | Seeking Feedback | Acknowledging Limitations | Growth Mindset</p>
            <p style="font-size: 0.9em; color: #666;">
                Ensure your microphone is enabled and use Chrome/Edge for best speech recognition results
            </p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    # Create public directory and copy STT script
    PUBLIC_DIR.mkdir(exist_ok=True)
    
    # Check if STT script exists, if not create a placeholder
    stt_script_path = PUBLIC_DIR / "stt.js"
    if not stt_script_path.exists():
        print("⚠️  STT script not found. Please ensure public/stt.js exists.")
        print("    You can copy the enhanced STT script to public/stt.js")
    
    # Create and launch the app
    demo = create_interface()
    
    print("🚀 Starting Humility Interview Assistant...")
    print("📋 Make sure the backend is running on http://127.0.0.1:8000")
    print("🎤 Speech-to-text will be available once the page loads")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False,
        debug=True,
        show_tips=True
    )