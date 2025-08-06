#!/usr/bin/env python3
"""
Humility Interview System - Integration Setup Script
This script sets up the complete system with enhanced STT functionality
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def setup_project():
    """Set up the project structure and files"""
    
    print("🚀 Setting up Humility Interview System...")
    
    # Get project root
    project_root = Path.cwd()
    print(f"📁 Project root: {project_root}")
    
    # Create directory structure
    directories = [
        "backend",
        "services", 
        "public",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create __init__.py files for Python packages
    init_files = [
        "backend/__init__.py",
        "services/__init__.py"
    ]
    
    for init_file in init_files:
        init_path = project_root / init_file
        init_path.touch()
        print(f"✅ Created: {init_file}")
    
    return project_root

def create_models_file(project_root: Path):
    """Create the models.py file for backend"""
    
    models_content = '''"""
Pydantic models for the Humility Interview API
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel

class GenerateAudioRequest(BaseModel):
    text: str

class GenerateAudioResponse(BaseModel):
    audio_b64: str

class AnalyzeRequest(BaseModel):
    transcript: str
    question: Optional[str] = ""

class AnalysisResponse(BaseModel):
    scores: Dict[str, Dict[str, Any]]

class TranscribeRequest(BaseModel):
    audio_b64: str

class TranscribeResponse(BaseModel):
    transcript: str
'''
    
    models_path = project_root / "backend" / "models.py"
    models_path.write_text(models_content)
    print("✅ Created backend/models.py")

def create_agent_manager(project_root: Path):
    """Create a simple agent manager for analysis"""
    
    agent_content = '''"""
Simple agent manager for humility analysis
"""

import asyncio
from typing import List, NamedTuple

class AnalysisResult(NamedTuple):
    agent_name: str
    score: float
    evidence: str

async def run_analysis_pipeline(transcript: str) -> List[AnalysisResult]:
    """
    Run analysis pipeline on transcript
    This is a simplified implementation - replace with your actual agents
    """
    
    results = []
    
    # Mock analysis agents
    agents = {
        "humility_detector": analyze_humility,
        "learning_mindset": analyze_learning,
        "feedback_seeking": analyze_feedback_seeking,
        "mistake_acknowledgment": analyze_mistakes
    }
    
    for agent_name, agent_func in agents.items():
        try:
            score, evidence = await agent_func(transcript)
            results.append(AnalysisResult(agent_name, score, evidence))
        except Exception as e:
            print(f"Agent {agent_name} failed: {e}")
            results.append(AnalysisResult(agent_name, 0.0, f"Analysis failed: {e}"))
    
    return results

async def analyze_humility(transcript: str) -> tuple:
    """Analyze humility indicators"""
    await asyncio.sleep(0.1)  # Simulate processing
    
    humble_words = ["learned", "mistake", "wrong", "help", "feedback", "improve"]
    score = sum(1 for word in humble_words if word.lower() in transcript.lower())
    score = min(5.0, score)  # Cap at 5
    
    evidence = f"Found {score} humility indicators in response"
    return score, evidence

async def analyze_learning(transcript: str) -> tuple:
    """Analyze learning mindset"""
    await asyncio.sleep(0.1)
    
    learning_words = ["learn", "grow", "develop", "understand", "realize", "discover"]
    score = sum(1 for word in learning_words if word.lower() in transcript.lower())
    score = min(5.0, score * 0.8)
    
    evidence = f"Learning mindset score based on {score/0.8:.0f} learning-related terms"
    return score, evidence

async def analyze_feedback_seeking(transcript: str) -> tuple:
    """Analyze feedback seeking behavior"""
    await asyncio.sleep(0.1)
    
    feedback_words = ["feedback", "advice", "guidance", "mentorship", "input", "perspective"]
    score = sum(1 for word in feedback_words if word.lower() in transcript.lower())
    score = min(5.0, score * 1.2)
    
    evidence = f"Feedback seeking indicators: {score/1.2:.0f} mentions"
    return score, evidence

async def analyze_mistakes(transcript: str) -> tuple:
    """Analyze mistake acknowledgment"""
    await asyncio.sleep(0.1)
    
    mistake_words = ["mistake", "error", "wrong", "failed", "struggled", "challenge"]
    score = sum(1 for word in mistake_words if word.lower() in transcript.lower())
    score = min(5.0, score * 1.0)
    
    evidence = f"Mistake acknowledgment: {score} relevant terms found"
    return score, evidence
'''
    
    agent_path = project_root / "backend" / "agent_manager.py"
    agent_path.write_text(agent_content)
    print("✅ Created backend/agent_manager.py")

def create_run_script(project_root: Path):
    """Create run scripts for easy startup"""
    
    # Backend run script
    backend_script = '''#!/bin/bash
echo "🚀 Starting Humility Interview Backend..."
echo "📋 Make sure you have installed dependencies:"
echo "   pip install fastapi uvicorn faster-whisper soundfile numpy"
echo ""

cd "$(dirname "$0")"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
'''
    
    backend_path = project_root / "run_backend.sh"
    backend_path.write_text(backend_script)
    backend_path.chmod(0o755)
    print("✅ Created run_backend.sh")
    
    # Frontend run script
    frontend_script = '''#!/bin/bash
echo "🎤 Starting Humility Interview Frontend..."
echo "📋 Make sure backend is running on http://127.0.0.1:8000"
echo ""

cd "$(dirname "$0")"
python gradio_app_enhanced.py
'''
    
    frontend_path = project_root / "run_frontend.sh"
    frontend_path.write_text(frontend_script)
    frontend_path.chmod(0o755)
    print("✅ Created run_frontend.sh")

def create_requirements_file(project_root: Path):
    """Create requirements.txt file"""
    
    requirements = '''# Core dependencies
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.0.0

# Audio processing
faster-whisper>=0.10.0
soundfile>=0.12.1
numpy>=1.24.0
librosa>=0.10.0

# TTS (optional but recommended)
TTS>=0.22.0

# Frontend
gradio>=4.0.0
streamlit>=1.28.0

# Utilities
requests>=2.31.0
python-multipart>=0.0.6
aiofiles>=23.0.0

# Development
pytest>=7.4.0
black>=23.0.0
'''
    
    req_path = project_root / "requirements.txt"
    req_path.write_text(requirements)
    print("✅ Created requirements.txt")

def create_readme(project_root: Path):
    """Create comprehensive README"""
    
    readme_content = '''# 🎤 Humility Interview System

AI-Powered Interview Practice with Real-time Speech Recognition and Analysis

## ✨ Features

- **Real-time Speech-to-Text**: Browser-based speech recognition with fallback to server-side STT
- **Text-to-Speech**: AI-generated question narration
- **Humility Analysis**: Multi-agent analysis of interview responses
- **Live Transcript**: Real-time transcription with confidence indicators
- **Export & Analysis**: Download transcripts and detailed analysis reports

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. System Requirements

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install -y espeak-ng ffmpeg

# For other systems, ensure you have:
# - espeak-ng (for TTS)
# - ffmpeg (for audio conversion)
```

### 3. Start the Backend

```bash
./run_backend.sh
# Or manually:
# python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start the Frontend

```bash
./run_frontend.sh  
# Or manually:
# python gradio_app_enhanced.py
```

### 5. Open in Browser

- Frontend: http://localhost:7860
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 🎯 Usage

1. **Enter Question**: Type or paste your interview question
2. **Listen**: Click "🔊 Speak Question" to hear the question
3. **Record**: Click "🎙️ Start Recording" and speak your answer
4. **Analyze**: Click "🔍 Analyze Response" for detailed feedback
5. **Export**: Download your session data for review

## 🔧 Configuration

### Audio Settings

The system supports multiple audio formats:
- Input: WAV, MP3, M4A, OGG, FLAC
- Output: WAV (16kHz, mono)

### Browser Compatibility

For best speech recognition results:
- ✅ Chrome/Chromium (recommended)
- ✅ Edge
- ✅ Safari
- ⚠️ Firefox (limited support)

## 📊 Analysis Agents

The system includes multiple analysis agents:

1. **Humility Detector**: Identifies humble language patterns
2. **Learning Mindset**: Evaluates growth-oriented responses  
3. **Feedback Seeking**: Assesses openness to feedback
4. **Mistake Acknowledgment**: Measures accountability

## 🛠️ Development

### Project Structure

```
humility-interview/
├── backend/
│   ├── main.py              # FastAPI backend
│   ├── models.py            # Pydantic models
│   └── agent_manager.py     # Analysis agents
├── services/
│   └── audio_processing.py  # STT/TTS services
├── public/
│   └── stt.js              # Enhanced STT client
├── gradio_app_enhanced.py   # Gradio frontend
├── requirements.txt         # Dependencies
├── run_backend.sh          # Backend startup
└── run_frontend.sh         # Frontend startup
```

### API Endpoints

- `POST /generate_audio`: Text-to-speech
- `POST /transcribe`: File-based STT
- `POST /transcribe_realtime`: Real-time STT
- `POST /analyze`: Response analysis
- `GET /system_info`: System status

## 🔍 Troubleshooting

### Common Issues

1. **Microphone not working**:
   - Ensure HTTPS is enabled (required for microphone access)
   - Check browser permissions
   - Try Chrome/Edge instead of Firefox

2. **STT not transcribing**:
   - Check if faster-whisper is installed
   - Verify audio format is supported
   - Check backend logs for errors

3. **TTS not working**:
   - Install Coqui-TTS: `pip install TTS`
   - Install espeak-ng system package
   - Check backend logs for model loading errors

### Debug Mode

Start with debug logging:
```bash
PYTHONPATH=. python -m uvicorn backend.main:app --log-level debug
```

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review backend logs
- Open an issue on GitHub
'''
    
    readme_path = project_root / "README.md"
    readme_path.write_text(readme_content)
    print("✅ Created README.md")

def check_dependencies():
    """Check if required system dependencies are installed"""
    
    print("\n🔍 Checking system dependencies...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check for system tools
    tools = {
        "ffmpeg": "Audio conversion",
        "espeak-ng": "Text-to-speech synthesis"
    }
    
    missing_tools = []
    for tool, purpose in tools.items():
        try:
            subprocess.run([tool, "--version"], 
                         capture_output=True, 
                         timeout=5)
            print(f"✅ {tool} - {purpose}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"⚠️ {tool} - {purpose} (optional but recommended)")
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"\n💡 To install missing tools on Ubuntu/Debian:")
        print(f"   sudo apt update && sudo apt install -y {' '.join(missing_tools)}")
    
    return len(missing_tools) == 0

def main():
    """Main setup function"""
    
    print("=" * 60)
    print("🎤 HUMILITY INTERVIEW SYSTEM - SETUP")
    print("=" * 60)
    
    try:
        # Setup project structure
        project_root = setup_project()
        
        # Create necessary files
        create_models_file(project_root)
        create_agent_manager(project_root)
        create_run_script(project_root)
        create_requirements_file(project_root)
        create_readme(project_root)
        
        # Copy the enhanced files to the project
        print("\n📥 Setting up enhanced files...")
        
        # Copy STT script to public directory
        stt_js_content = '''// Enhanced STT script content will be here
// This is a placeholder - copy the actual stt.js content from the artifacts
console.log("STT script loaded - replace with actual enhanced STT code");
'''
        stt_path = project_root / "public" / "stt.js"
        stt_path.write_text(stt_js_content)
        print("✅ Created public/stt.js (placeholder)")
        
        # Check system dependencies
        deps_ok = check_dependencies()
        
        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETE!")
        print("=" * 60)
        
        print("\n📋 Next Steps:")
        print("1. Install Python dependencies:")
        print("   pip install -r requirements.txt")
        
        if not deps_ok:
            print("\n2. Install system dependencies:")
            print("   sudo apt update && sudo apt install -y espeak-ng ffmpeg")
        
        print("\n3. Replace placeholder files with actual enhanced code:")
        print("   - Copy enhanced STT script to public/stt.js")
        print("   - Update backend/main.py with enhanced backend code")
        print("   - Update services/audio_processing.py with improved audio service")
        print("   - Update gradio_app_enhanced.py with enhanced Gradio app")
        
        print("\n4. Start the system:")
        print("   ./run_backend.sh   # Terminal 1")
        print("   ./run_frontend.sh  # Terminal 2")
        
        print("\n5. Open in browser:")
        print("   http://localhost:7860")
        
        print("\n🎯 The system will provide:")
        print("   ✅ Real-time speech-to-text")
        print("   ✅ AI question narration")
        print("   ✅ Humility analysis")
        print("   ✅ Session export")
        
        print("\n📞 Need help? Check README.md for troubleshooting")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())