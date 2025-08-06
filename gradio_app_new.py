# import gradio as gr
# import requests
# import base64
# import json
# import os
# import tempfile
# import time
# import wave
# from datetime import datetime
# from pathlib import Path

# # Backend URL
# BACKEND = "http://127.0.0.1:8000"
# SAMPLE_RATE = 24000  # Match this with your TTS model's sample rate

# def synthesize(question: str):
#     """Convert text to speech using the backend TTS service."""
#     if not question.strip():
#         return None, "❌ Please enter a question first."
#     try:
#         response = requests.post(
#             f"{BACKEND}/generate_audio",
#             json={"text": question},
#             timeout=30
#         )
#         response.raise_for_status()
#         audio_data = response.json().get("audio_b64")
#         if not audio_data:
#             return None, "❌ No audio data received from the server."
#         try:
#             # Return as file path for better compatibility
#             wav_bytes = base64.b64decode(audio_data)
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#                 f.write(wav_bytes)
#                 temp_path = f.name
#             return temp_path, ""
#         except Exception as e:
#             return None, f"❌ Error processing audio data: {str(e)}"
#     except requests.exceptions.RequestException as e:
#         return None, f"❌ Error: Could not connect to TTS service. {str(e)}"
#     except (KeyError, json.JSONDecodeError) as e:
#         return None, f"❌ Error processing TTS response. {str(e)}"


# def analyze_response(question: str, transcript: str) -> str:
#     """Analyze the transcript for humility indicators."""
#     if not transcript.strip():
#         return "⚠️ Please provide a transcript to analyze."
#     try:
#         response = requests.post(
#             f"{BACKEND}/analyze",
#             json={"question": question, "transcript": transcript},
#             timeout=60
#         )
#         response.raise_for_status()
#         report = response.json().get("scores", {})
#         if not report:
#             return "❌ No analysis results received."
            
#         # Format the report as markdown
#         markdown = "## Analysis Results\n\n"
#         for category, details in report.items():
#             markdown += f"### {category.replace('_', ' ').title()}\n"
#             markdown += f"**Score**: {details.get('score', 'N/A')}/5\n"
#             if 'evidence' in details and details['evidence']:
#                 markdown += f"**Evidence**: {details['evidence']}\n"
#             if 'suggestion' in details and details['suggestion']:
#                 markdown += f"**Suggestion**: {details['suggestion']}\n"
#             markdown += "\n"
            
#         return markdown
        
#     except requests.exceptions.RequestException as e:
#         return f"❌ Error: Could not connect to analysis service. {str(e)}"
#     except (KeyError, json.JSONDecodeError) as e:
#         return f"❌ Error processing analysis response. {str(e)}"


# def get_audio_duration(audio_file: str) -> float:
#     """Get the duration of an audio file in seconds."""
#     try:
#         with wave.open(audio_file, 'rb') as wav:
#             frames = wav.getnframes()
#             rate = wav.getframerate()
#             duration = frames / float(rate)
#             return duration
#     except Exception as e:
#         print(f"Error getting audio duration: {e}")
#         return 0.0

# # Create public directory if it doesn't exist
# os.makedirs("public", exist_ok=True)

import os
import gradio as gr
from pathlib import Path

# Create public directory if it doesn't exist
PUBLIC_DIR = Path("public")
PUBLIC_DIR.mkdir(exist_ok=True)

# Create the Gradio interface
with gr.Blocks(title="AI Interview Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 AI Interview Assistant
    Practice your interview skills with AI-powered feedback.
    """)
    
    # Add custom CSS and JS for the interface
    gr.HTML("""
    <script src="/file=public/stt.js"></script>
    <script>
    // Dispatch gradio_loaded event when the page is fully loaded
    document.addEventListener('DOMContentLoaded', () => {
        setTimeout(() => {
            document.dispatchEvent(new Event('gradio_loaded'));
            console.log('Gradio loaded event dispatched');
        }, 1000);
    });
    </script>
    <style>
        .gradio-container { max-width: 900px !important; }
        .gradio-button { margin: 0 5px; }
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            background: #f9fafb;
            border-radius: 10px;
            margin: 10px 0;
            font-size: 0.9rem;
            color: #6b7280;
        }
        #recordingIndicator {
            display: none;
            align-items: center;
            gap: 10px;
            margin: 10px 0;
            padding: 12px 16px;
            background: rgba(239, 68, 68, 0.1);
            border-radius: 8px;
            border-left: 4px solid #ef4444;
            font-size: 0.95em;
            color: #ef4444;
            animation: fadeIn 0.3s ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-5px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .pulse {
            width: 12px;
            height: 12px;
            background: #ef4444;
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }
        .word-count {
            font-weight: 600;
            color: #4f46e5;
        }
        #stt-controls {
            margin: 15px 0;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
        }
        /* Button styles */
        /* Custom button styles */
        .gradio-button {
            display: inline-flex !important;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin: 4px !important;
            padding: 10px 20px !important;
            border: none !important;
            border-radius: 8px !important;
            cursor: pointer;
            font-weight: 500 !important;
            font-size: 0.95em !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1) !important;
            min-width: 120px;
            height: 40px;
        }
        
        /* Primary button (green) */
        .gradio-button.primary {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%) !important;
            color: white !important;
        }
        
        /* Secondary button (blue) */
        .gradio-button.secondary {
            background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%) !important;
            color: white !important;
        }
        
        /* Danger button (red) */
        .gradio-button.danger {
            background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%) !important;
            color: white !important;
        }
        
        /* Success button (green) */
        .gradio-button.success {
            background: linear-gradient(135deg, #4CAF50 0%, #388E3C 100%) !important;
            color: white !important;
        }
        
        .gradio-button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        }
        
        .gradio-button:active {
            transform: translateY(1px) !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
        }
        #startBtn {
            background: #10b981 !important;
            color: white !important;
        }
        #startBtn:hover {
            background: #0d9f6e !important;
            transform: translateY(-1px);
        }
        #startBtn:disabled {
            background: #9ca3af !important;
            cursor: not-allowed;
        }
        #stopBtn {
            background: #ef4444 !important;
            color: white !important;
            display: none;
        }
        #stopBtn:hover {
            background: #dc2626 !important;
            transform: translateY(-1px);
        }
        #stopBtn:disabled {
            background: #9ca3af !important;
            cursor: not-allowed;
        }
        #clearBtn {
            background: #6b7280 !important;
            color: white !important;
        }
        #clearBtn:hover {
            background: #4b5563 !important;
            transform: translateY(-1px);
        }
    </style>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Question section
            question = gr.Textbox(
                label="Question",
                placeholder="Enter your interview question here...",
                lines=3,
                elem_id="question"
            )
            
            # Speech controls
            with gr.Row():
                start_btn = gr.Button("🎤 Start Recording", elem_id="startBtn")
                stop_btn = gr.Button("⏹ Stop Recording", elem_id="stopBtn", visible=False)
                clear_btn = gr.Button("🗑 Clear", elem_id="clearBtn")
            
            # Recording indicator
            with gr.Row():
                recording_indicator = gr.HTML("""
                    <div id="recordingIndicator" style="display: none;">
                        <div class="pulse"></div>
                        <span>Recording in progress...</span>
                    </div>
                """)
            
            # Language selection
            with gr.Row():
                language = gr.Dropdown(
                    label="Language",
                    choices=[
                        ("English (US)", "en-US"),
                        ("English (UK)", "en-GB"),
                        ("Spanish", "es-ES"),
                        ("French", "fr-FR"),
                        ("German", "de-DE")
                    ],
                    value="en-US",
                    elem_id="languageSelect"
                )
            # Transcript section
            with gr.Row():
                transcript = gr.Textbox(
                    label="Your Response",
                    placeholder="Your spoken response will appear here...",
                    lines=8,
                    elem_id="transcript",
                    interactive=True
                )
            
            # Word count display
            with gr.Row():
                word_count = gr.HTML("""
                    <div style="display: flex; justify-content: space-between; width: 100%; color: #6b7280; font-size: 0.9em;">
                        <div>Words: <span id="wordCount" class="word-count">0</span></div>
                        <div id="recordingTime">00:00</div>
                    </div>
                """)
            
            # Analysis section
            with gr.Row():
                analyze_btn = gr.Button("📊 Analyze Response", variant="primary")
            
            with gr.Row():
                analysis_output = gr.Markdown("")
            
            # Export controls
            with gr.Row():
                export_btn = gr.Button("💾 Export Session")
                copy_btn = gr.Button("📋 Copy Transcript")
    
    # Event handlers
    def toggle_recording(recording):
        return not recording, gr.update(visible=not recording), gr.update(visible=recording)
    
    def clear_transcript():
        return ""
    
    def analyze_response(question_text, transcript_text):
        if not transcript_text.strip():
            return "Please record or enter a response before analyzing."
        
        # This is a placeholder - replace with actual analysis logic
        return f"## Analysis for: {question_text[:50]}...\n\nThis is a placeholder analysis. The actual analysis would go here."
    
    def export_session(question_text, transcript_text, analysis_text):
        if not transcript_text.strip():
            return "No transcript to export."
        
        # Create a downloadable text file
        timestamp = gr.utils.get_datetime_for_filename()
        content = f"# Interview Session - {timestamp}\n\n"
        content += f"## Question\n{question_text}\n\n"
        content += f"## Your Response\n{transcript_text}\n\n"
        
        if analysis_text:
            content += f"## Analysis\n{analysis_text}"
        
        return content
    
    # Connect buttons to functions
    start_btn.click(
        fn=toggle_recording,
        inputs=[gr.State(False)],
        outputs=[gr.State(True), start_btn, stop_btn]
    )
    
    stop_btn.click(
        fn=toggle_recording,
        inputs=[gr.State(True)],
        outputs=[gr.State(False), stop_btn, start_btn]
    )
    
    clear_btn.click(
        fn=clear_transcript,
        outputs=transcript
    )
    
    analyze_btn.click(
        fn=analyze_response,
        inputs=[question, transcript],
        outputs=analysis_output
    )
    
    export_btn.click(
        fn=export_session,
        inputs=[question, transcript, analysis_output],
        outputs=gr.File(label="Download Session")
    )
    
    copy_btn.click(
        fn=lambda x: x,
        inputs=transcript,
        outputs=gr.Textbox(visible=False)
    )

# Run the app
if __name__ == "__main__":
    # Ensure the public directory exists
    PUBLIC_DIR.mkdir(exist_ok=True)
    
    # Copy the STT script to the public directory if it doesn't exist
    stt_js_path = PUBLIC_DIR / "stt.js"
    if not stt_js_path.exists():
        with open(stt_js_path, "w") as f:
            f.write("""
            // Speech recognition initialization will be handled by the frontend
            console.log("STT script loaded");
            
            // This is a placeholder - the actual STT implementation is in the HTML
            window.speechToText = {
                start: () => console.log("Start recording"),
                stop: () => console.log("Stop recording"),
                setLanguage: (lang) => console.log("Set language to", lang)
            };
            """)
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
            
#             # Action buttons row
#             with gr.Row():
#                 speak_btn = gr.Button("🔊 Speak Question", variant="primary")
#                 analyze_btn = gr.Button("📊 Analyze Response", variant="secondary")
            
#             # Microphone permission status
#             with gr.Group():
#                 gr.HTML("""
#                 <div class="permission-status">
#                     <h4 style="margin-top: 0; margin-bottom: 10px;">🎤 Microphone Status</h4>
#                     <div id="permissionStatus">
#                         <div style="background: #fef3c7; color: #92400e; padding: 15px; border-radius: 10px; border-left: 4px solid #f59e0b;">
#                             ⏳ Checking microphone status...
#                         </div>
#                     </div>
#                     <div id="micPermissionHelper" style="display: none; margin-top: 10px; padding: 10px; background: #f8fafc; border-radius: 6px; border: 1px dashed #cbd5e1;">
#                         <p style="margin: 0 0 10px 0; font-size: 0.9em;">
#                             <strong>Microphone Access Required</strong><br>
#                             Please allow microphone access in your browser to use voice recording.
#                         </p>
#                         <button class="gradio-button" onclick="requestMicrophoneAccess()" style="padding: 6px 12px; font-size: 0.9em;">
#                             Grant Microphone Access
#                         </button>
#                     </div>
#                 </div>
#                 """)
            
#             # Recording controls
#             with gr.Group():
#                 gr.HTML("<h4 style='margin-bottom: 5px;'>🎙️ Recording Controls</h4>")
#                 with gr.Row():
#                     start_btn = gr.Button("🎤 Start Recording", elem_id="startBtn")
#                     stop_btn = gr.Button("⏹️ Stop", elem_id="stopBtn", interactive=False)
#                     clear_btn = gr.Button("🧹 Clear", elem_id="clearBtn")
                
#                 # Recording indicator
#                 gr.HTML("""
#                 <div id="recordingIndicator" class="recording-indicator" style="display: none;">
#                     <div class="recording-dot"></div>
#                     <span id="statusText">Ready</span>
#                 </div>
#                 """)
                
#                 # Status and word count
#                 gr.HTML("""
#                 <div class="status-bar">
#                     <div>Status: <span id="statusText">Ready</span></div>
#                     <div>Words: <span id="wordCount" class="word-count">0</span></div>
#                 </div>
#                 """)
            
#             # Language selection
#             with gr.Group():
#                 language = gr.Dropdown(
#                     choices=[
#                         ("English (US)", "en-US"),
#                         ("English (UK)", "en-GB"),
#                         ("Spanish", "es-ES"),
#                         ("French", "fr-FR"),
#                         ("German", "de-DE"),
#                         ("Hindi", "hi-IN"),
#                         ("Japanese", "ja-JP"),
#                         ("Chinese (Simplified)", "zh-CN")
#                     ],
#                     value="en-US",
#                     label="Recognition Language",
#                     interactive=True,
#                     elem_id="languageSelect"
#                 )
            
#             # Messages area
#             gr.HTML("""
#             <div id="errorMessage" class="error-message"></div>
#             <div id="successMessage" class="success-message"></div>
#             """)
            
#             # Transcript area (right column)
#             with gr.Column(scale=2):
#                 # Transcript area
#                 transcript = gr.Textbox(
#                     label="Your Response",
#                     placeholder="Speak your answer here...",
#                     lines=15,
#                     show_label=True,
#                     elem_id="transcript"
#                 )
                
#                 # Analysis results
#                 with gr.Group():
#                     gr.Markdown("### Analysis Results")
#                     analysis = gr.Markdown("Your analysis will appear here...", elem_id="analysisResults")
                
#                 # Debug console (initially hidden, can be shown if needed)
#                 with gr.Accordion("Debug Console", open=False):
#                     debug_console = gr.Textbox(
#                         label="Debug Info",
#                         lines=6,
#                         interactive=False,
#                         elem_id="debug-info"
#                     )
    
#     # Add audio output component (hidden, as we'll handle audio through the browser)
#     audio_output = gr.Audio(visible=False)
    
#     # Connect speak button
#     speak_btn.click(
#         fn=synthesize,
#         inputs=[question],
#         outputs=[audio_output, analysis]
#     )
    
#     # Connect analyze button
#     analyze_btn.click(
#         fn=analyze_response,
#         inputs=[question, transcript],
#         outputs=[analysis]
#     )
    
#     # Initialize the app when the page loads
#     demo.load(
#         None,
#         None,
#         None,
#         js="""
#         function() {
#             if (window.initializeApp) {
#                 initializeApp();
#             } else {
#                 console.log('Waiting for app to load...');
#                 setTimeout(arguments.callee, 100);
#             }
#             return [];
#         }
#         """
#     )
    
#     # Add custom CSS for the interface
#     custom_css = """
#     <style>
#     /* Status indicators */
#     .status {
#         padding: 5px 10px;
#         border-radius: 4px;
#         font-weight: 500;
#         margin: 5px 0;
#     }
#     .status.ready {
#         background-color: #e0f2fe;
#         color: #0369a1;
#     }
#     .status.listening {
#         background-color: #dcfce7;
#         color: #166534;
#         animation: pulse 1.5s infinite;
#     }
#     .status.error {
#         background-color: #fee2e2;
#         color: #991b1b;
#     }
    
#     /* Recording indicator */
#     .recording-indicator {
#         display: flex;
#         align-items: center;
#         gap: 8px;
#         margin: 10px 0;
#         padding: 8px 12px;
#         background-color: #f8fafc;
#         border-radius: 6px;
#         border-left: 4px solid #ef4444;
#     }
    
#     .recording-dot {
#         width: 12px;
#         height: 12px;
#         background-color: #ef4444;
#         border-radius: 50%;
#         animation: pulse 1.5s infinite;
#     }
    
#     /* Buttons */
#     button {
#         transition: all 0.2s ease-in-out;
#     }
    
#     button:disabled {
#         opacity: 0.6;
#         cursor: not-allowed;
#     }
    
#     /* Status bar */
#     .status-bar {
#         display: flex;
#         justify-content: space-between;
#         margin-top: 10px;
#         padding: 8px 12px;
#         background-color: #f8fafc;
#         border-radius: 6px;
#         font-size: 0.9em;
#         color: #475569;
#     }
    
#     .word-count {
#         font-weight: 600;
#         color: #1e40af;
#     }
    
#     /* Animations */
#     @keyframes pulse {
#         0% { opacity: 1; }
#         50% { opacity: 0.5; }
#         100% { opacity: 1; }
#     }
    
#     /* Error and success messages */
#     .error-message {
#         display: none;
#         padding: 10px 15px;
#         margin: 10px 0;
#         background-color: #fee2e2;
#         color: #991b1b;
#         border-radius: 6px;
#         border-left: 4px solid #dc2626;
#     }
    
#     .success-message {
#         display: none;
#         padding: 10px 15px;
#         margin: 10px 0;
#         background-color: #dcfce7;
#         color: #166534;
#         border-radius: 6px;
#         border-left: 4px solid #16a34a;
#     }
    
#     /* Permission helper */
#     .permission-helper {
#         margin-top: 10px;
#         padding: 12px;
#         background-color: #f8fafc;
#         border: 1px dashed #cbd5e1;
#         border-radius: 6px;
#         font-size: 0.9em;
#     }
    
#     .permission-helper button {
#         margin-top: 8px;
#         padding: 6px 12px;
#         background-color: #3b82f6;
#         color: white;
#         border: none;
#         border-radius: 4px;
#         cursor: pointer;
#     }
    
#     .permission-helper button:hover {
#         background-color: #2563eb;
#     }
#     </style>
#     """
    
#     # Add the custom CSS and JavaScript to the page
#     demo.css = custom_css
    
#     # Add the JavaScript for speech recognition
#     stt_js = """
#     <script>
#     // Global variables
#     let recognition;
#     let isRecording = false;
#     let finalTranscript = '';
#     let interimTranscript = '';
    
#     // Initialize speech recognition
#     function initializeSpeechRecognition() {
#         const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        
#         if (!SpeechRecognition) {
#             console.error('Speech recognition not supported in this browser');
#             updateStatus('Speech recognition not supported', 'error');
#             return null;
#         }
        
#         recognition = new SpeechRecognition();
#         recognition.continuous = true;
#         recognition.interimResults = true;
        
#         recognition.onstart = function() {
#             console.log('Speech recognition started');
#             isRecording = true;
#             updateStatus('Listening...', 'listening');
#             document.getElementById('startBtn').disabled = true;
#             document.getElementById('stopBtn').disabled = false;
#             document.getElementById('recordingIndicator').style.display = 'flex';
#         };
        
#         recognition.onresult = function(event) {
#             interimTranscript = '';
            
#             for (let i = event.resultIndex; i < event.results.length; i++) {
#                 const transcript = event.results[i][0].transcript;
#                 if (event.results[i].isFinal) {
#                     finalTranscript += transcript + ' ';
#                 } else {
#                     interimTranscript += transcript;
#                 }
#             }
            
#             // Update the transcript textarea
#             const transcriptTextarea = document.querySelector('textarea[data-testid="textbox"]');
#             if (transcriptTextarea) {
#                 transcriptTextarea.value = finalTranscript + interimTranscript;
#                 // Trigger input event to update Gradio's state
#                 const inputEvent = new Event('input', { bubbles: true });
#                 transcriptTextarea.dispatchEvent(inputEvent);
#             }
            
#             updateWordCount();
#         };
        
#         recognition.onerror = function(event) {
#             console.error('Speech recognition error:', event.error);
#             updateStatus('Error: ' + event.error, 'error');
#             stopRecording();
#         };
        
#         recognition.onend = function() {
#             console.log('Speech recognition ended');
#             if (isRecording) {
#                 // If we're still supposed to be recording, restart recognition
#                 recognition.start();
#             } else {
#                 document.getElementById('startBtn').disabled = false;
#                 document.getElementById('stopBtn').disabled = true;
#                 document.getElementById('recordingIndicator').style.display = 'none';
#             }
#         };
        
#         return recognition;
#     }
    
#     // Set up event listeners
#     function setupEventListeners() {
#         const startBtn = document.getElementById('startBtn');
#         const stopBtn = document.getElementById('stopBtn');
#         const clearBtn = document.getElementById('clearBtn');
#         const languageSelect = document.getElementById('languageSelect');
        
#         if (startBtn) {
#             startBtn.addEventListener('click', startRecording);
#         }
        
#         if (stopBtn) {
#             stopBtn.addEventListener('click', stopRecording);
#         }
        
#         if (clearBtn) {
#             clearBtn.addEventListener('click', clearTranscript);
#         }
        
#         if (languageSelect) {
#             languageSelect.addEventListener('change', function() {
#                 if (recognition) {
#                     recognition.lang = this.value;
#                     console.log('Language changed to:', this.value);
#                 }
#             });
#         }
#     }
    
#     // Start recording
#     async function startRecording() {
#         // Check if already recording
#         if (isRecording) {
#             console.log('Already recording');
#             return;
#         }

#         try {
#             // Request microphone permission again to be sure
#             const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
#             // Stop all tracks to release the microphone
#             stream.getTracks().forEach(track => track.stop());
            
#             // Initialize recognition if not already done
#             if (!recognition) {
#                 recognition = initializeSpeechRecognition();
#                 if (!recognition) return;
#             }
            
#             // Reset transcripts
#             finalTranscript = '';
#             interimTranscript = '';
            
#             // Get the selected language
#             const languageSelect = document.getElementById('languageSelect');
#             if (languageSelect) {
#                 recognition.lang = languageSelect.value;
#             }
            
#             // Update UI
#             document.getElementById('startBtn').disabled = true;
#             document.getElementById('stopBtn').disabled = false;
#             document.getElementById('recordingIndicator').style.display = 'flex';
            
#             // Set recording state
#             isRecording = true;
            
#             // Start recognition
#             try {
#                 recognition.start();
#                 console.log('Starting speech recognition');
#                 updateStatus('Listening... Speak now.', 'recording');
#             } catch (startError) {
#                 console.error('Error starting recognition:', startError);
#                 updateStatus('Error starting speech recognition: ' + startError.message, 'error');
#                 isRecording = false;
#                 document.getElementById('startBtn').disabled = false;
#                 document.getElementById('stopBtn').disabled = true;
#                 document.getElementById('recordingIndicator').style.display = 'none';
#             }
            
#         } catch (error) {
#             console.error('Error accessing microphone:', error);
#             updateStatus('Error: Could not access microphone. Please check your browser permissions.', 'error');
#             isRecording = false;
#             document.getElementById('startBtn').disabled = false;
#             document.getElementById('stopBtn').disabled = true;
#             document.getElementById('recordingIndicator').style.display = 'none';
#         }
#     }  
    
#     // Stop recording
#     function stopRecording() {
#         if (!isRecording) {
#             console.log('Not currently recording');
#             return;
#         }
        
#         isRecording = false;
        
#         try {
#             if (recognition) {
#                 recognition.stop();
#                 console.log('Stopped speech recognition');
#             }
            
#             // Update UI
#             document.getElementById('startBtn').disabled = false;
#             document.getElementById('stopBtn').disabled = true;
#             document.getElementById('recordingIndicator').style.display = 'none';
            
#             if (finalTranscript.trim() === '') {
#                 updateStatus('Recording stopped. No speech was detected.', 'warning');
#             } else {
#                 updateStatus('Recording stopped. You can analyze your response now.', 'ready');
#             }
            
#         } catch (error) {
#             console.error('Error stopping recognition:', error);
#             updateStatus('Error stopping recording: ' + error.message, 'error');
#         }
        
#         // Force update the UI in case the recognition didn't trigger onend
#         setTimeout(() => {
#             if (!isRecording) {
#                 document.getElementById('startBtn').disabled = false;
#                 document.getElementById('stopBtn').disabled = true;
#                 document.getElementById('recordingIndicator').style.display = 'none';
#             }
#         }, 1000);
#     }  
    
#     // Clear transcript
#     function clearTranscript() {
#         const transcriptTextarea = document.querySelector('textarea[data-testid="textbox"]');
#         if (transcriptTextarea) {
#             transcriptTextarea.value = '';
#             // Trigger input event to update Gradio's state
#             const inputEvent = new Event('input', { bubbles: true });
#             transcriptTextarea.dispatchEvent(inputEvent);
#         }
#         finalTranscript = '';
#         interimTranscript = '';
#         updateWordCount();
#     }
    
#     // Update status message
#     function updateStatus(message, type = '') {
#         const statusElement = document.getElementById('statusText');
#         if (statusElement) {
#             statusElement.textContent = message;
#             statusElement.className = type;
#         }
#     }
    
#     // Update word count
#     function updateWordCount() {
#         const wordCountElement = document.getElementById('wordCount');
#         if (wordCountElement) {
#             const text = finalTranscript + interimTranscript;
#             const wordCount = text.trim() === '' ? 0 : text.trim().split(/\s+/).length;
#             wordCountElement.textContent = wordCount;
#         }
#     }
    
#     // Initialize the app when the page loads
#     document.addEventListener('DOMContentLoaded', async function() {
#         console.log('DOM fully loaded');
        
#         // Check for browser support first
#         if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
#             updateStatus('Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.', 'error');
#             return;
#         }
        
#         // Request microphone permission
#         try {
#             const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
#             // Stop all tracks to release the microphone
#             stream.getTracks().forEach(track => track.stop());
#             console.log('Microphone access granted');
#             updateStatus('Microphone is ready. Click Start Recording to begin.', 'ready');
#         } catch (err) {
#             console.error('Microphone access denied:', err);
#             updateStatus('Microphone access is required for recording. Please allow microphone access and refresh the page.', 'error');
#             return;
#         }
        
#         setupEventListeners();
        
#         // Set initial button states
#         document.getElementById('stopBtn').disabled = true;
#         document.getElementById('recordingIndicator').style.display = 'none';
#     });
    
#     // Expose functions to the global scope
#     window.startRecording = startRecording;
#     window.stopRecording = stopRecording;
#     </script>
#     """
    
#     # Add the JavaScript to the page
#     demo.js = stt_js
    
#     # Check browser support
#     function checkBrowserSupport() {
#         if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
#             showError('Your browser does not support speech recognition. Please use Chrome, Edge, or Safari.');
#             return false;
#         }
        

#         if (!('speechSynthesis' in window)) {
#             showError('Your browser does not support speech synthesis.');
#             return false;
#         }

#         return true;
#     }

#     // Setup speech recognition
#     function setupSpeechRecognition() {
#         const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
#         recognition = new SpeechRecognition();

#         // Configure recognition settings
#         recognition.continuous = true;
#         recognition.interimResults = true;
#         recognition.maxAlternatives = 3;
#         recognition.lang = document.getElementById('languageSelect').value;

#         // Event handlers
#         recognition.onstart = handleRecognitionStart;
#         recognition.onresult = handleRecognitionResult;
#         recognition.onerror = handleRecognitionError;
#         recognition.onend = handleRecognitionEnd;
#     }

#     // Speech recognition event handlers
#     function handleRecognitionStart() {
#         isRecording = true;
#         updateUI();
#         updateStatus('Listening...');
#         showSuccess('Recording started successfully!');
#     }

#     function handleRecognitionResult(event) {
#         interimTranscript = '';
        
#         for (let i = event.resultIndex; i < event.results.length; i++) {
#             const result = event.results[i];
#             const transcript = result[0].transcript;
#             const confidence = result[0].confidence;
            
#             if (result.isFinal) {
#                 finalTranscript += transcript + ' ';
#             } else {
#                 interimTranscript += transcript;
#             }
#         }
        
#         updateTranscriptDisplay();
#         updateWordCount();
#     }

#     function handleRecognitionError(event) {
#         console.error('Speech recognition error:', event);
#         let errorMsg = 'An error occurred during speech recognition: ';
        
#         switch(event.error) {
#             case 'no-speech':
#                 errorMsg += 'No speech detected. Please try speaking clearly.';
#                 break;
#             case 'audio-capture':
#                 errorMsg += 'No microphone found or microphone access denied.';
#                 showMicHelper();
#                 break;
#             case 'not-allowed':
#                 errorMsg += 'Microphone access denied. Please allow microphone access.';
#                 showMicHelper();
#                 break;
#             case 'network':
#                 errorMsg += 'Network error occurred. Please check your connection.';
#                 break;
#             default:
#                 errorMsg += event.error;
#         }
        
#         showError(errorMsg);
#         stopRecording();
#     }

#     function handleRecognitionEnd() {
#         if (isRecording) {
#             // Auto-restart if we're still supposed to be recording
#             setTimeout(() => {
#                 if (isRecording) {
#                     recognition.start();
#                 }
#             }, 100);
#         } else {
#             updateStatus('Ready');
#             updateUI();
#         }
#     }

#     // Recording control functions
#     async function startRecording() {
#         if (!checkBrowserSupport()) return;
        
#         // First, request microphone permission explicitly
#         try {
#             await requestMicrophoneAccess();
#             recognition.lang = document.getElementById('languageSelect').value;
#             recognition.start();
#             hideMessages();
#         } catch (error) {
#             showError('Failed to start recording: ' + error.message);
#             checkMicrophoneStatus(); // Show detailed status
#         }
#     }

#     // Check microphone status and permissions
#     async function checkMicrophoneStatus() {
#         const statusDiv = document.getElementById('permissionStatus');
        
#         try {
#             // Check if getUserMedia is supported
#             if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
#                 statusDiv.innerHTML = `
#                     <div style="background: #fef2f2; color: #dc2626; padding: 15px; border-radius: 10px; border-left: 4px solid #dc2626;">
#                         ❌ <strong>Not Supported:</strong> Your browser doesn't support microphone access. Please use Chrome, Firefox, Safari, or Edge with HTTPS.
#                     </div>`;
#                 return;
#             }

#             // Check current permission state
#             const permission = await navigator.permissions.query({ name: 'microphone' });
            
#             let statusHTML = '';
#             switch (permission.state) {
#                 case 'granted':
#                     statusHTML = `
#                         <div style="background: #f0fdf4; color: #16a34a; padding: 15px; border-radius: 10px; border-left: 4px solid #16a34a;">
#                             ✅ <strong>Microphone Access:</strong> Granted! You should be able to record.
#                         </div>`;
                    
#                     // Test actual microphone access
#                     try {
#                         const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
#                         stream.getTracks().forEach(track => track.stop());
#                         statusHTML += `
#                             <div style="background: #f0fdf4; color: #16a34a; padding: 15px; border-radius: 10px; border-left: 4px solid #16a34a; margin-top: 10px;">
#                                 ✅ <strong>Microphone Test:</strong> Working perfectly!
#                             </div>`;
#                     } catch (testError) {
#                         statusHTML += `
#                             <div style="background: #fef2f2; color: #dc2626; padding: 15px; border-radius: 10px; border-left: 4px solid #dc2626; margin-top: 10px;">
#                                 ❌ <strong>Microphone Test Failed:</strong> ${testError.message}
#                             </div>`;
#                     }
#                     break;
                    
#                 case 'denied':
#                     statusHTML = `
#                         <div style="background: #fef2f2; color: #dc2626; padding: 15px; border-radius: 10px; border-left: 4px solid #dc2626;">
#                             ❌ <strong>Permission Denied:</strong> Microphone access has been blocked.
#                             <br><strong>Fix:</strong> Click the 🎤 or 🔒 icon in your address bar and allow microphone access.
#                         </div>`;
#                     break;
                    
#                 case 'prompt':
#                     statusHTML = `
#                         <div style="background: #fef3c7; color: #92400e; padding: 15px; border-radius: 10px; border-left: 4px solid #f59e0b;">
#                             ⏳ <strong>Permission Required:</strong> Browser will ask for microphone access.
#                             <br><button class="gradio-button" onclick="requestMicrophoneAccess()" style="margin-top: 10px;">Request Access Now</button>
#                         </div>`;
#                     break;
#             }
            
#             statusDiv.innerHTML = statusHTML;
            
#         } catch (error) {
#             statusDiv.innerHTML = `
#                 <div style="background: #fef2f2; color: #dc2626; padding: 15px; border-radius: 10px; border-left: 4px solid #dc2626;">
#                     ❌ <strong>Error checking permissions:</strong> ${error.message}
#                 </div>`;
#         }
#     }

#     // Direct microphone access request
#     async function requestMicrophoneAccess() {
#         try {
#             const stream = await navigator.mediaDevices.getUserMedia({ 
#                 audio: {
#                     echoCancellation: true,
#                     noiseSuppression: true,
#                     autoGainControl: true
#                 }
#             });
#             stream.getTracks().forEach(track => track.stop());
#             showSuccess('✅ Microphone access granted! You can now start recording.');
#             checkMicrophoneStatus(); // Refresh status
#             return true;
#         } catch (error) {
#             let errorMessage = '';
#             switch (error.name) {
#                 case 'NotAllowedError':
#                     errorMessage = 'Permission denied. Please click the microphone icon in your address bar and select "Allow".';
#                     break;
#                 case 'NotFoundError':
#                     errorMessage = 'No microphone found. Please connect a microphone and try again.';
#                     break;
#                 case 'NotSupportedError':
#                     errorMessage = 'Microphone not supported. Please use HTTPS and a modern browser.';
#                     break;
#                 default:
#                     errorMessage = error.message;
#             }
#             showError('❌ Failed to access microphone: ' + errorMessage);
#             throw new Error(errorMessage);
#         }
#     }

#     function stopRecording() {
#         if (recognition && isRecording) {
#             isRecording = false;
#             recognition.stop();
#             updateStatus('Processing...');
#             setTimeout(() => {
#                 updateStatus('Ready');
#                 updateUI();
#             }, 1000);
#         }
#     }

#     // Text-to-speech functions
#     function speakQuestion() {
#         const questionText = document.getElementById('question').value.trim();
        
#         if (!questionText) {
#             showError('Please enter a question to speak.');
#             return;
#         }

#         if (currentUtterance) {
#             speechSynthesis.cancel();
#         }

#         currentUtterance = new SpeechSynthesisUtterance(questionText);
#         currentUtterance.rate = 0.9;
#         currentUtterance.pitch = 1;
#         currentUtterance.volume = 1;

#         currentUtterance.onstart = () => {
#             showSuccess('Speaking question...');
#         };

#         currentUtterance.onend = () => {
#             showSuccess('Question spoken. You can now start recording your answer.');
#             currentUtterance = null;
#         };

#         currentUtterance.onerror = (event) => {
#             showError('Error speaking question: ' + event.error);
#             currentUtterance = null;
#         };

#         speechSynthesis.speak(currentUtterance);
#     }

#     function stopSpeaking() {
#         if (speechSynthesis.speaking) {
#             speechSynthesis.cancel();
#             currentUtterance = null;
#             showSuccess('Speech stopped.');
#         }
#     }

#     // Transcript management functions
#     function updateTranscriptDisplay() {
#         const transcriptTextarea = document.querySelector('textarea[data-testid="textbox"]');
#         if (!transcriptTextarea) return;
        
#         // Update the textarea with both final and interim transcripts
#         transcriptTextarea.value = finalTranscript + (interimTranscript ? '<span class="interim-text">' + interimTranscript + '</span>' : '');
        
#         // Trigger input event to update Gradio's state
#         const inputEvent = new Event('input', { bubbles: true });
#         transcriptTextarea.dispatchEvent(inputEvent);
        
#         // Also update the hidden input that Gradio uses for form submission
#         const hiddenInput = transcriptTextarea.previousElementSibling;
#         if (hiddenInput && hiddenInput.tagName === 'INPUT' && hiddenInput.type === 'hidden') {
#             hiddenInput.value = finalTranscript + interimTranscript;
#         }
#     }

#     function clearTranscript() {
#         finalTranscript = '';
#         interimTranscript = '';
#         updateTranscriptDisplay();
#         updateWordCount();
#         showSuccess('Transcript cleared.');
#     }

#     function updateWordCount() {
#         // Split on any whitespace character
#         // Split on any whitespace character using a character class
#     const wordCount = finalTranscript.trim().split(/[\s\u00A0]+/).filter(word => word.length > 0).length;
#         const wordCountElement = document.getElementById('wordCount');
#         if (wordCountElement) {
#             wordCountElement.textContent = wordCount;
#         }
#     }

#     // UI helper functions
#     function updateUI() {
#         const startBtn = document.getElementById('startBtn');
#         const stopBtn = document.getElementById('stopBtn');
#         const recordingIndicator = document.getElementById('recordingIndicator');
#         const languageSelect = document.getElementById('languageSelect');

#         if (isRecording) {
#             if (startBtn) startBtn.disabled = true;
#             if (stopBtn) stopBtn.disabled = false;
#             if (recordingIndicator) recordingIndicator.style.display = 'flex';
#             if (languageSelect) languageSelect.disabled = true;
#         } else {
#             if (startBtn) startBtn.disabled = false;
#             if (stopBtn) stopBtn.disabled = true;
#             if (recordingIndicator) recordingIndicator.style.display = 'none';
#             if (languageSelect) languageSelect.disabled = false;
#         }
#     }

#     function updateStatus(status) {
#         const statusElement = document.getElementById('statusText');
#         if (statusElement) {
#             statusElement.textContent = status;
#         }
#     }

#     function showError(message) {
#         const errorDiv = document.getElementById('errorMessage');
#         if (errorDiv) {
#             errorDiv.textContent = message;
#             errorDiv.style.display = 'block';
#             hideSuccess();
#             setTimeout(() => {
#                 errorDiv.style.display = 'none';
#             }, 5000);
#         }
#     }

#     function showSuccess(message) {
#         const successDiv = document.getElementById('successMessage');
#         if (successDiv) {
#             successDiv.textContent = message;
#             successDiv.style.display = 'block';
#             hideError();
#             setTimeout(() => {
#                 successDiv.style.display = 'none';
#             }, 3000);
#         }
#     }

#     function hideError() {
#         const errorDiv = document.getElementById('errorMessage');
#         if (errorDiv) errorDiv.style.display = 'none';
#     }

#     function hideSuccess() {
#         const successDiv = document.getElementById('successMessage');
#         if (successDiv) successDiv.style.display = 'none';
#     }

#     function hideMessages() {
#         hideError();
#         hideSuccess();
#     }

#     function showMicHelper() {
#         const micHelper = document.getElementById('micPermissionHelper');
#         if (micHelper) micHelper.style.display = 'block';
#     }

#     function hideMicHelper() {
#         const micHelper = document.getElementById('micPermissionHelper');
#         if (micHelper) micHelper.style.display = 'none';
#     }

#     // Event listeners
#     document.addEventListener('DOMContentLoaded', function() {
#         // Initialize the app
#         initializeApp();
        
#         // Add event listeners for buttons
#         const startBtn = document.getElementById('startBtn');
#         const stopBtn = document.getElementById('stopBtn');
#         const clearBtn = document.getElementById('clearBtn');
#         const ttsBtn = document.querySelector('[aria-label="🔊 Speak Question"]');
#         const languageSelect = document.getElementById('languageSelect');
        
#         if (startBtn) {
#             startBtn.addEventListener('click', startRecording);
#         }
        
#         if (stopBtn) {
#             stopBtn.addEventListener('click', stopRecording);
#         }
        
#         if (clearBtn) {
#             clearBtn.addEventListener('click', clearTranscript);
#         }
        
#         if (ttsBtn) {
#             ttsBtn.addEventListener('click', speakQuestion);
#         }
        
#         if (languageSelect) {
#             languageSelect.addEventListener('change', function() {
#                 if (recognition) {
#                     recognition.lang = this.value;
#                 }
#             });
#         }
        
#         // Add keyboard shortcuts
#         document.addEventListener('keydown', function(event) {
#             // Ctrl+Alt+R to start/stop recording
#             if (event.ctrlKey && event.altKey && event.key === 'r') {
#                 event.preventDefault();
#                 if (isRecording) {
#                     stopRecording();
#                 } else {
#                     startRecording();
#                 }
#             }
            
#             // Ctrl+Alt+C to clear transcript
#             if (event.ctrlKey && event.altKey && event.key === 'c') {
#                 event.preventDefault();
#                 clearTranscript();
#             }
#         });
#     });
#     """
    
#     # Add the JavaScript to the page using a custom HTML component
#     stt_html = gr.HTML(f"""
#     <script>
#     {stt_js}
#     </script>
#     """)
    
#     # Speak button is already connected above
    
#     analyze_btn.click(
#         fn=analyze_response,
#         inputs=[question, transcript],
#         outputs=[analysis]
#     )


# # if __name__ == "__main__":
# #     demo.launch(
# #         server_name="0.0.0.0",
# #         server_port=7860,
# #         ssl_keyfile="key.pem",
# #         ssl_certfile="cert.pem",
# #         show_error=True,
# #         share=False
# #     )

# if __name__ == "__main__":
#     # Run the Gradio app directly
#     demo.launch(
#         server_name="localhost",
#         server_port=7860,
#         show_error=True,
#         share=False
#     )





















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
        server_name="localhost",
        server_port=7860,
        show_error=True,
        share=False,
        debug=True,
        #show_tips=True
    )