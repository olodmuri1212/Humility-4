#!/usr/bin/env python3
"""
Test script to verify audio capture and processing functionality.
This will help diagnose audio issues in the interview application.
"""

import asyncio
import base64
import requests
import wave
import tempfile
import os
from pathlib import Path

def test_audio_generation():
    """Test if the backend can generate audio from text."""
    print("🔊 Testing audio generation...")
    
    try:
        # Test the backend TTS endpoint
        response = requests.post(
            "http://localhost:8000/generate_audio",
            json={"text": "Hello, this is a test of the audio generation system."},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "audio_b64" in data:
                audio_bytes = base64.b64decode(data["audio_b64"])
                
                # Save to a test file
                test_file = "test_generated_audio.wav"
                with open(test_file, "wb") as f:
                    f.write(audio_bytes)
                
                print(f"✅ Audio generation successful!")
                print(f"   - Audio length: {len(audio_bytes)} bytes")
                print(f"   - Saved to: {test_file}")
                print(f"   - Play with: aplay {test_file}")
                
                return True
            else:
                print("❌ No audio data in response")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing audio generation: {e}")
        return False

def test_audio_transcription():
    """Test if the backend can transcribe audio."""
    print("\n🎤 Testing audio transcription...")
    
    # Check if we have a test audio file
    test_files = ["test_generated_audio.wav", "test_audio.wav"]
    test_file = None
    
    for file in test_files:
        if os.path.exists(file):
            test_file = file
            break
    
    if not test_file:
        print("⚠️  No test audio file found. Generate audio first or create test_audio.wav")
        return False
    
    try:
        # Read the audio file
        with open(test_file, "rb") as f:
            audio_data = f.read()
        
        # Convert to base64
        audio_b64 = base64.b64encode(audio_data).decode()
        
        # Test the backend STT endpoint
        response = requests.post(
            "http://localhost:8000/transcribe",
            json={"audio_b64": audio_b64},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if "transcript" in data:
                print(f"✅ Audio transcription successful!")
                print(f"   - Transcript: '{data['transcript']}'")
                return True
            else:
                print("❌ No transcript in response")
                return False
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing audio transcription: {e}")
        return False

def test_system_audio():
    """Test system audio capabilities."""
    print("\n🔧 Testing system audio capabilities...")
    
    # Test if espeak is available
    try:
        import subprocess
        result = subprocess.run(["espeak-ng", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ espeak-ng is available")
        else:
            print("❌ espeak-ng not working properly")
    except Exception as e:
        print(f"❌ espeak-ng test failed: {e}")
    
    # Test if we can generate a simple audio file
    try:
        test_cmd = [
            "espeak-ng", 
            "-s", "150",  # Speed
            "-v", "en",   # Voice
            "-w", "system_test.wav",  # Output file
            "System audio test"
        ]
        result = subprocess.run(test_cmd, timeout=10)
        if result.returncode == 0 and os.path.exists("system_test.wav"):
            print("✅ System audio generation works")
            print("   - Test file: system_test.wav")
            print("   - Play with: aplay system_test.wav")
        else:
            print("❌ System audio generation failed")
    except Exception as e:
        print(f"❌ System audio test failed: {e}")

def main():
    """Run all audio tests."""
    print("🧪 AUDIO SYSTEM TEST")
    print("=" * 50)
    
    # Check if backend is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend server is running")
        else:
            print("❌ Backend server responded with error")
            return
    except Exception:
        print("❌ Backend server is not running!")
        print("   Start it with: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000")
        return
    
    print()
    
    # Run tests
    tts_ok = test_audio_generation()
    stt_ok = test_audio_transcription()
    test_system_audio()
    
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY:")
    print(f"   🔊 TTS (Text-to-Speech): {'✅ PASS' if tts_ok else '❌ FAIL'}")
    print(f"   🎤 STT (Speech-to-Text): {'✅ PASS' if stt_ok else '❌ FAIL'}")
    
    if tts_ok and stt_ok:
        print("\n🎉 All audio tests passed! Your system should work.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main() 