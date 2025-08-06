// Speech Recognition Wrapper for Gradio App
class SpeechToText {
    constructor() {
        this.recognition = null;
        this.isRecording = false;
        this.finalTranscript = '';
        this.interimTranscript = '';
        this.onResultCallback = null;
        this.onErrorCallback = null;
        this.initialize();
    }

    // Initialize speech recognition
    initialize() {
        try {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {
                throw new Error('Speech recognition not supported in this browser');
            }

            this.recognition = new SpeechRecognition();
            this.recognition.continuous = true;
            this.recognition.interimResults = true;
            this.recognition.lang = 'en-US';

            // Event handlers
            this.recognition.onstart = () => {
                this.isRecording = true;
                console.log('Speech recognition started');
            };

            this.recognition.onresult = (event) => {
                this.interimTranscript = '';
                
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        this.finalTranscript += transcript + ' ';
                    } else {
                        this.interimTranscript += transcript;
                    }
                }

                if (this.onResultCallback) {
                    this.onResultCallback({
                        final: this.finalTranscript,
                        interim: this.interimTranscript
                    });
                }
            };

            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                if (this.onErrorCallback) {
                    this.onErrorCallback(event.error);
                }
            };

            this.recognition.onend = () => {
                if (this.isRecording) {
                    // Restart recognition if still supposed to be recording
                    this.recognition.start();
                }
            };

        } catch (error) {
            console.error('Error initializing speech recognition:', error);
            throw error;
        }
    }

    // Start speech recognition
    start() {
        if (!this.recognition) {
            throw new Error('Speech recognition not initialized');
        }

        if (this.isRecording) {
            console.warn('Already recording');
            return;
        }

        this.finalTranscript = '';
        this.interimTranscript = '';
        
        try {
            this.recognition.start();
            return true;
        } catch (error) {
            console.error('Error starting speech recognition:', error);
            throw error;
        }
    }

    // Stop speech recognition
    stop() {
        if (!this.recognition) {
            return;
        }

        this.isRecording = false;
        
        try {
            this.recognition.stop();
        } catch (error) {
            console.error('Error stopping speech recognition:', error);
        }
    }

    // Set language
    setLanguage(lang) {
        if (this.recognition) {
            this.recognition.lang = lang;
        }
    }

    // Set callbacks
    onResult(callback) {
        this.onResultCallback = callback;
    }

    onError(callback) {
        this.onErrorCallback = callback;
    }

    // Get current transcript
    getTranscript() {
        return {
            final: this.finalTranscript,
            interim: this.interimTranscript,
            combined: this.finalTranscript + (this.interimTranscript ? ' ' + this.interimTranscript : '')
        };
    }

    // Check if speech recognition is supported
    static isSupported() {
        return 'webkitSpeechRecognition' in window || 'SpeechRecognition' in window;
    }

    // Request microphone permission
    static async requestMicrophonePermission() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            // Stop all tracks to release the microphone
            stream.getTracks().forEach(track => track.stop());
            return true;
        } catch (error) {
            console.error('Microphone permission denied:', error);
            return false;
        }
    }
}

// Initialize global STT instance
let speechToText = null;

// Function to initialize speech recognition
document.addEventListener('DOMContentLoaded', async () => {
    if (!SpeechToText.isSupported()) {
        console.error('Speech recognition not supported in this browser');
        return;
    }

    try {
        // Request microphone permission
        const hasPermission = await SpeechToText.requestMicrophonePermission();
        if (!hasPermission) {
            console.error('Microphone permission denied');
            return;
        }

        // Initialize speech recognition
        speechToText = new SpeechToText();
        console.log('Speech recognition initialized');

        // Set up event listeners for Gradio components
        setupGradioIntegration();
    } catch (error) {
        console.error('Error initializing speech recognition:', error);
    }
});

// Set up Gradio integration
function setupGradioIntegration() {
    // This function will be called when Gradio components are ready
    document.addEventListener('gradio_loaded', () => {
        console.log('Gradio loaded, setting up STT integration');
        
        // Find the transcript textarea
        const transcriptTextarea = document.querySelector('textarea[data-testid="textbox"]');
        if (!transcriptTextarea) {
            console.error('Could not find transcript textarea');
            return;
        }

        // Set up result callback
        speechToText.onResult(({ final, interim }) => {
            const fullText = final + (interim ? ' ' + interim : '');
            transcriptTextarea.value = fullText;
            
            // Trigger input event to update Gradio's state
            const inputEvent = new Event('input', { bubbles: true });
            transcriptTextarea.dispatchEvent(inputEvent);
            
            // Also update the hidden input that Gradio uses for form submission
            const hiddenInput = transcriptTextarea.previousElementSibling;
            if (hiddenInput && hiddenInput.tagName === 'INPUT') {
                hiddenInput.value = fullText;
                hiddenInput.dispatchEvent(new Event('input', { bubbles: true }));
            }
        });

        // Set up error callback
        speechToText.onError((error) => {
            console.error('Speech recognition error:', error);
            // You can update the UI to show the error to the user
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = `Error: ${error}`;
            document.body.appendChild(errorDiv);
            
            // Remove the error message after 5 seconds
            setTimeout(() => {
                errorDiv.remove();
            }, 5000);
        });

        // Find and set up record/stop buttons
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');

        if (startBtn) {
            startBtn.addEventListener('click', () => {
                try {
                    speechToText.start();
                    console.log('Recording started');
                    
                    // Update UI
                    startBtn.disabled = true;
                    if (stopBtn) stopBtn.disabled = false;
                    
                    // Show recording indicator
                    const indicator = document.getElementById('recordingIndicator');
                    if (indicator) indicator.style.display = 'flex';
                    
                } catch (error) {
                    console.error('Error starting recording:', error);
                }
            });
        }

        if (stopBtn) {
            stopBtn.addEventListener('click', () => {
                speechToText.stop();
                console.log('Recording stopped');
                
                // Update UI
                if (startBtn) startBtn.disabled = false;
                stopBtn.disabled = true;
                
                // Hide recording indicator
                const indicator = document.getElementById('recordingIndicator');
                if (indicator) indicator.style.display = 'none';
            });
        }

        // Set up language selector
        const languageSelect = document.getElementById('languageSelect');
        if (languageSelect) {
            languageSelect.addEventListener('change', (e) => {
                speechToText.setLanguage(e.target.value);
                console.log('Language set to:', e.target.value);
            });
        }
    });
}

// Export for global access
window.SpeechToText = SpeechToText;
window.speechToText = speechToText;
