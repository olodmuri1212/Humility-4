
<<<<<<< HEAD
AI-Powered Interview Practice with Real-time Speech Recognition and Analysis

## ✨ Features

- **Real-time Speech-to-Text**: Browser-based Web Speech API for instant transcription with fallback to server-side STT
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

### Modes of Operation

1. **Auto Mode (Record Audio)**
   - Record your voice directly in the browser
   - Automatic transcription and analysis
   - Best for: Hands-free operation

2. **Speech-to-Text Mode (Beta)**
   - Real-time transcription using Web Speech API
   - See your words appear as you speak
   - Best for: Immediate feedback on speech clarity

3. **Manual Mode**
   - Type or paste your response
   - Best for: Reviewing or editing responses

### Basic Workflow

1. **Enter Question**: Type or paste your interview question
2. **Listen**: Click "🔊 Speak Question" to hear the question
3. **Respond**: 
   - For Auto Mode: Click "🎙️ Start Recording" and speak
   - For Speech-to-Text: Click "🎤 Start Recording" and speak
   - For Manual: Type your response
4. **Analyze**: Click "🔍 Analyze Response" for detailed feedback
5. **Export**: Download your session data for review

## 🎤 Web Speech API Integration

The application now includes browser-based speech recognition using the Web Speech API, which provides:

- **Real-time transcription**: See your words appear as you speak
- **No server processing**: All speech recognition happens in your browser
- **Faster response**: Immediate feedback without waiting for server processing
- **Offline capability**: Works without an internet connection (after initial page load)

### Using Web Speech API

1. Click the "Speech-to-Text (Beta)" radio button
2. Click "🎤 Start Recording" when ready to speak
3. Speak clearly into your microphone
4. View real-time transcription
5. Click "⏹️ Stop Recording" when finished
6. Click "Analyze Response" to get feedback

## 🔧 Configuration

### Audio Settings

The system supports multiple audio formats:
- Input: WAV, MP3, M4A, OGG, FLAC
- Output: WAV (16kHz, mono)

### Browser Compatibility

For best speech recognition results:
- ✅ Chrome/Chromium (recommended) - Full Web Speech API support
- ✅ Edge - Full Web Speech API support
- ✅ Safari - Full Web Speech API support (desktop only)
- ⚠️ Firefox - Limited Web Speech API support

> **Note**: For the best experience with Web Speech API, use the latest version of Chrome or Edge on a desktop computer.

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
# Humility-3
=======
>>>>>>> be1121112650ad4a953f9aad159b975d9ba7e31c
# Humility-2
# Humility-2
# Humility-2
# Humility-3
