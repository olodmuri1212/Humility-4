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
        #show_tips=True
    )