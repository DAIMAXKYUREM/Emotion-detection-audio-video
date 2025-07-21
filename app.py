# app.py - Final Version

import asyncio
import base64
import cv2
import numpy as np
import logging
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from PIL import Image
from io import BytesIO
import threading

# Import your model classes
from face_model import FacialEmotionDetector
from speech_model import SpeechEmotionAnalyzer
from text_model import TextSummarizer

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO)

# This line is critical for Gunicorn to find the application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*")

# --- Model Initialization (runs once on startup) ---
try:
    print("Initializing models...")
    face_detector = FacialEmotionDetector(model_path='best.pt')
    speech_analyzer = SpeechEmotionAnalyzer()
    text_summarizer = TextSummarizer()
    print("\nAll models initialized successfully!")
except Exception as e:
    logging.error(f"Fatal error during model initialization: {e}")
    face_detector = None
    speech_analyzer = None
    text_summarizer = None

# --- Global State Management ---
last_facial_emotion = "neutral"
processing_lock = threading.Lock()

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handles a new client connection."""
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handles a client disconnection."""
    print('Client disconnected')

@socketio.on('video_frame')
def handle_video_frame(data_url):
    """
    Receives a video frame from the client, processes it for facial emotion,
    and sends back the annotated frame.
    """
    global last_facial_emotion
    if not face_detector:
        return

    try:
        # Decode the base64 image data
        header, encoded = data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Detect emotion
        annotated_frame, detected_emotion = face_detector.detect_emotion(frame)
        if detected_emotion:
            last_facial_emotion = detected_emotion

        # Encode the annotated frame back to base64 to send to the client
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        b64_frame = base64.b64encode(buffer).decode('utf-8')
        
        emit('annotated_frame', f'data:image/jpeg;base64,{b64_frame}')

    except Exception as e:
        logging.error(f"Error processing video frame: {e}")


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """
    Receives an audio chunk, analyzes it, and sends back the summary.
    This function now handles the full analysis pipeline.
    """
    if not speech_analyzer or not text_summarizer:
        emit('analysis_result', {'summary': 'Error: Backend models not loaded.'})
        return

    # Prevent multiple requests from processing at the same time
    if not processing_lock.acquire(blocking=False):
        print("Processor is busy. Skipping audio chunk.")
        return

    try:
        print("Received audio chunk for processing...")
        # The audio data from the browser is raw Float32Array bytes
        audio_data = np.frombuffer(data, dtype=np.float32)
        
        # 1. Get transcription and vocal emotion
        transcribed_text, vocal_emotion = speech_analyzer.process_audio(audio_data, sample_rate=16000)
        
        print(f"Transcription: '{transcribed_text}'")
        print(f"Vocal Emotion: {vocal_emotion} | Last Facial Emotion: {last_facial_emotion}")

        if not transcribed_text:
            summary = "Could not understand audio. Please speak more clearly."
        else:
            # Immediately notify the user that the heavy processing is starting.
            emit('analysis_result', {'summary': 'Generating AI summary... (this may take a moment on CPU)'})
            
            # 2. Get summary from Llama 3
            summary = text_summarizer.summarize_with_context(
                transcribed_text,
                last_facial_emotion if last_facial_emotion else "unknown",
                vocal_emotion if vocal_emotion else "unknown"
            )
        
        print(f"AI Summary: {summary}")
        
        # Send the final result back to the client
        emit('analysis_result', {'summary': summary})

    except Exception as e:
        logging.error(f"Error during audio analysis: {e}")
        emit('analysis_result', {'summary': 'An error occurred during analysis.'})
    finally:
        # Release the lock so the next chunk can be processed
        processing_lock.release()

if __name__ == '__main__':
    print("Starting Flask-SocketIO server...")
    # This is for local development. For production, use a Gunicorn server.
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
