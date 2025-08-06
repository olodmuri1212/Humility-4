#!/usr/bin/env python3
"""
Simple microphone test to verify audio recording works.
This will record 5 seconds of audio and save it to a file.
"""

import pyaudio
import wave
import time

def test_microphone_recording():
    """Record 5 seconds of audio from the default microphone."""
    print("🎤 Testing microphone recording...")
    
    # Audio recording parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5
    OUTPUT_FILE = "mic_test.wav"
    
    try:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        # Check for available input devices
        print("📋 Available audio input devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"   {i}: {info['name']}")
        
        print(f"\n🔴 Recording {RECORD_SECONDS} seconds of audio...")
        print("   Speak into your microphone now!")
        
        # Open recording stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        frames = []
        
        # Record audio
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            
            # Progress indicator
            if i % 10 == 0:
                print(".", end="", flush=True)
        
        print(f"\n✅ Recording completed!")
        
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save the audio to a WAV file
        with wave.open(OUTPUT_FILE, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        
        print(f"📁 Audio saved to: {OUTPUT_FILE}")
        print(f"🔊 Play with: paplay {OUTPUT_FILE}")
        
        # Get file size for verification
        import os
        file_size = os.path.getsize(OUTPUT_FILE)
        print(f"📊 File size: {file_size} bytes")
        
        if file_size > 1000:  # At least 1KB
            print("✅ Recording appears successful!")
            return True
        else:
            print("❌ Recording file is too small - may have failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during microphone test: {e}")
        return False

def main():
    """Run the microphone test."""
    print("🧪 MICROPHONE TEST")
    print("=" * 50)
    
    # Test microphone recording
    success = test_microphone_recording()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Microphone test passed!")
        print("   Your microphone is working correctly.")
        print("   You can now test the WebRTC recording in the interview app.")
    else:
        print("⚠️  Microphone test failed.")
        print("   Check your microphone permissions and connections.")

if __name__ == "__main__":
    main() 