Real-Time Multi-Modal Emotion Analyzer This project is a web-based "social interpreter" designed to help individuals, particularly those with autism, better understand emotional cues in social interactions. It uses a combination of AI models to provide real-time analysis of facial expressions, vocal tone, and spoken language, delivering a concise summary of the emotional context.

This project was developed for the SuPrathon 2k25 National Level Virtual Hackathon (Problem ID: SP25-AM04).

Live Demo: https://huggingface.co/spaces/ShReYas6969/Suprathon

Features Real-Time Facial Emotion Detection: Uses a YOLOv11n model to identify faces and classify their emotions (e.g., happy, sad, neutral) directly from a live webcam feed.

Vocal Emotion Analysis: Analyzes the tone of voice from microphone input to detect the underlying emotion in speech.

Accurate Speech-to-Text: Employs OpenAI's Whisper model to transcribe spoken words into text.

Context-Aware AI Summarization: Uses Meta's Llama 3 model to generate a simple, one-sentence summary of what was said, taking into account the detected facial and vocal emotions.

Web-Based Interface: A clean and simple frontend built with HTML and JavaScript that runs in any modern web browser.

Real-Time Communication: Utilizes WebSockets for low-latency communication between the browser and the backend server.

Tech Stack Backend: Python, Flask, Flask-SocketIO

Web Server: Gunicorn

Frontend: HTML, JavaScript, Socket.IO Client

AI & Machine Learning:

Facial Emotion: Ultralytics YOLOv8, OpenCV

Speech Transcription: openai-whisper

Vocal Emotion: transformers (Hugging Face)

Text Summarization: transformers (for Llama 3 on CPU), unsloth (for Llama 3 on GPU)

Core Library: PyTorch

Deployment: Docker, Hugging Face Spaces

Project Setup and Installation Prerequisites Python 3.9+

pip for package management

A Hugging Face account

Clone the Repository git clone https://github.com/DAIMAXKYUREM/Emotion-detection-audio-video cd Emotion-detection-audio-video

Install Dependencies Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

Obtain Necessary Files and Secrets Facial Detection Model: You need a trained YOLO model file named best.pt. Place this file in the root directory of the project.
Hugging Face Access:

Request Access to Llama 3: Go to the Meta Llama 3 8B Instruct model page and accept the terms to get access.

Get an Access Token: Go to your Hugging Face settings and create a new read token.

Create a .env file: In the root directory, create a file named .env and add your token to it:

HF_TOKEN="your_hugging_face_token_goes_here"

Note: The .env file is included in .gitignore and should never be committed to your repository.

Run the Application Locally Start the Flask development server by running app.py:
python app.py

Open your web browser and navigate to http://127.0.0.1:5000.

Deployment on Hugging Face Spaces This application is designed to be deployed using Docker on Hugging Face Spaces.

Create a New Space: Create a new Space on Hugging Face, selecting Docker as the SDK and choosing a Blank template. A CPU instance (like CPU upgrade - 16GB RAM) is sufficient, though a GPU will provide much better performance.

Upload Files: Upload all the project files and folders (app.py, face_model.py, speech_model.py, text_model.py, requirements.txt, Dockerfile, best.pt, and the templates/ folder).

Add Secret Token: In the Space's Settings tab, go to Repository secrets and add a new secret:

Name: HF_TOKEN

Value: Paste your Hugging Face read token here.

Build and Run: Hugging Face will automatically build the Docker image from the Dockerfile and start the application. The live link will be available on the Space's main page.

Project Structure . ├── app.py # Main Flask application and WebSocket server ├── face_model.py # Class for facial emotion detection ├── speech_model.py # Class for speech-to-text and vocal emotion analysis ├── text_model.py # Class for Llama 3 text summarization ├── best.pt # Trained YOLO model for facial emotion ├── requirements.txt # Python dependencies ├── Dockerfile # Docker configuration for deployment ├── .gitignore # Specifies files for Git to ignore (e.g., .env) └── templates/ └── index.html # Frontend HTML and JavaScript

Team - AstralNomads 
Tanmayyash Mallick - Machine Learning
Shreyas Chander - Machine Learning
Alok Ranjan Tripathy - Machine Learning
Nilay Agarwal - Backend & Deployment
---
title: Suprathon
emoji: 🐨
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: A real-time emotion analyzer to understand social cues.
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
