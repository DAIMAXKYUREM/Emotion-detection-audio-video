# main.py

import cv2
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import threading
import queue
import time
import os

# Import the model classes from the other files
from face_model import FacialEmotionDetector
from speech_model import SpeechEmotionAnalyzer
from text_model import TextSummarizer

# --- Configuration ---
SAMPLE_RATE = 16000  # Sample rate for audio recording (16k is standard for speech models)
CHANNELS = 1  # Mono audio
SILENCE_THRESHOLD = 0.01  # Amplitude threshold to detect silence
SILENCE_DURATION = 1.5  # Seconds of silence to mark the end of a sentence
AUDIO_CHUNK_SIZE = 1024  # Number of frames per buffer

# --- Global Variables ---
audio_queue = queue.Queue()
is_recording = True


def audio_callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status)
    audio_queue.put(indata.copy())


def record_audio_thread():
    """
    A thread that continuously records audio from the microphone
    and puts the data into a queue.
    """
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        blocksize=AUDIO_CHUNK_SIZE, callback=audio_callback):
        print("Audio recording thread started. Recording...")
        while is_recording:
            time.sleep(0.1)
    print("Audio recording thread finished.")


def main():
    global is_recording

    # --- Model Initialization ---
    try:
        print("Initializing models...")
        # IMPORTANT: Make sure you have the 'yolov11n.pt' model file in this directory.
        face_detector = FacialEmotionDetector(model_path='yolov11n.pt')
        speech_analyzer = SpeechEmotionAnalyzer()
        text_summarizer = TextSummarizer()
        print("\nAll models initialized successfully!")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please make sure the required model files are present.")
        return
    except Exception as e:
        print(f"An error occurred during model initialization: {e}")
        return

    # --- Start Audio Recording ---
    audio_thread = threading.Thread(target=record_audio_thread)
    audio_thread.daemon = True
    audio_thread.start()

    # --- Start Webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # --- Main Application Loop ---
    print("\nApplication running. Speak into the microphone. Press 'q' to quit.")

    sentence_audio = []
    silence_start_time = None
    last_facial_emotion = "neutral"

    try:
        while True:
            # 1. Process Video Frame
            ret, frame = cap.read()
            if not ret:
                break

            # Detect facial emotion
            annotated_frame, detected_emotion = face_detector.detect_emotion(frame)
            if detected_emotion:
                last_facial_emotion = detected_emotion

            cv2.imshow('Real-time Analysis', annotated_frame)

            # 2. Process Audio Stream
            audio_buffer = []
            while not audio_queue.empty():
                audio_buffer.append(audio_queue.get())

            if audio_buffer:
                # Concatenate all audio chunks from the queue
                current_audio_chunk = np.concatenate(audio_buffer)
                sentence_audio.append(current_audio_chunk)

                # Check for silence
                if np.abs(current_audio_chunk).mean() < SILENCE_THRESHOLD:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time > SILENCE_DURATION:
                        # Silence detected, process the sentence
                        full_sentence_audio = np.concatenate(sentence_audio)

                        print("\n" + "=" * 50)
                        print("End of sentence detected. Processing...")

                        # Process in a separate thread to avoid freezing the webcam feed
                        processing_thread = threading.Thread(
                            target=process_and_summarize,
                            args=(full_sentence_audio, speech_analyzer, text_summarizer, last_facial_emotion)
                        )
                        processing_thread.start()

                        # Reset for the next sentence
                        sentence_audio = []
                        silence_start_time = None
                else:
                    # Sound detected, reset silence timer
                    silence_start_time = None

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Cleanup
        print("Shutting down...")
        is_recording = False
        audio_thread.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()


def process_and_summarize(audio_data, speech_analyzer, text_summarizer, facial_emotion):
    """
    Function to run in a thread for processing audio and generating a summary.
    """
    # 1. Get transcription and vocal emotion
    transcribed_text, vocal_emotion = speech_analyzer.process_audio(audio_data, SAMPLE_RATE)

    print(f"Transcription: '{transcribed_text}'")
    print(f"Vocal Emotion: {vocal_emotion} | Facial Emotion: {facial_emotion}")

    if not transcribed_text:
        print("Summary: Could not transcribe audio.")
        print("=" * 50 + "\n")
        return

    # 2. Get summary from Llama 3
    summary = text_summarizer.summarize_with_context(
        transcribed_text,
        facial_emotion if facial_emotion else "unknown",
        vocal_emotion if vocal_emotion else "unknown"
    )

    print(f"\nAI Summary: {summary}")
    print("=" * 50 + "\n")


if __name__ == '__main__':
    main()
