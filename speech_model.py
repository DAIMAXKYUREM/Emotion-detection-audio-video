# speech_model.py

import whisper
from transformers import pipeline
import numpy as np
import os
from typing import Union, Tuple

class SpeechEmotionAnalyzer:
    """
    A class to transcribe audio and classify the emotion from the speech.
    """
    def __init__(self, whisper_model="tiny", emotion_model="prithivMLmods/Speech-Emotion-Classification"):
        """
        Initializes the SpeechEmotionAnalyzer.

        Args:
            whisper_model (str): The name of the Whisper model to use for transcription.
            emotion_model (str): The Hugging Face model to use for speech emotion classification.
        """
        # Load the Whisper model for speech-to-text
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model(whisper_model)
        
        # Load the pipeline for audio classification
        print("Loading speech emotion classification model...")
        self.emotion_classifier = pipeline(
            "audio-classification",
            model=emotion_model
        )
        print("SpeechEmotionAnalyzer initialized successfully.")

    def process_audio(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[str, Union[str, None]]:
        """
        Transcribes audio and classifies its emotion.

        Args:
            audio_data (np.ndarray): The raw audio data as a NumPy array.
            sample_rate (int): The sample rate of the audio data.

        Returns:
            A tuple containing:
                - The transcribed text.
                - The detected emotion label (e.g., 'SAD', 'HAPPY') or None if classification fails.
        """
        # Ensure audio is in the correct format (float32) for Whisper
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / 32767.0

        # 1. Transcribe audio to text using Whisper
        print("Transcribing audio...")
        transcription_result = self.whisper_model.transcribe(audio_data)
        text = transcription_result.get("text", "").strip()
        
        # 2. Classify emotion from the audio
        print("Classifying speech emotion...")
        try:
            # The pipeline expects a dictionary with 'raw' audio data and 'sampling_rate'
            audio_input = {"raw": audio_data, "sampling_rate": sample_rate}
            emotion_results = self.emotion_classifier(audio_input, top_k=1)
            
            # The result is a list of lists, get the top result
            if emotion_results and emotion_results[0]:
                emotion = emotion_results[0][0]['label']
            else:
                emotion = None
        except Exception as e:
            print(f"Could not classify speech emotion: {e}")
            emotion = None
            
        return text, emotion


if __name__ == '__main__':
    # Example usage: This part is harder to test standalone without an audio file.
    # The main.py script will handle live microphone input.
    # You can uncomment and modify the following to test with a local audio file.
    
    # from scipy.io.wavfile import read
    # try:
    #     analyzer = SpeechEmotionAnalyzer()
    #     # Make sure you have a 'test_audio.wav' file in the same directory.
    #     sample_rate, audio_data = read("test_audio.wav")
    #     text, emotion = analyzer.process_audio(audio_data, sample_rate)
    #     print("--- Analysis Result ---")
    #     print(f"Transcription: {text}")
    #     print(f"Vocal Emotion: {emotion}")
    # except FileNotFoundError:
    #     print("Could not find 'test_audio.wav'. Skipping standalone test.")
    # except Exception as e:
    #     print(f"An error occurred during standalone test: {e}")
    pass
