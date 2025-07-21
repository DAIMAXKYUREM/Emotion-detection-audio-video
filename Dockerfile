# Use a more complete Python base image to ensure more system libraries are present
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create a dedicated, non-root user to run the application
RUN useradd --create-home appuser

# Create a dedicated, writable directory for all cache files.
# Set environment variables to tell all libraries to use this directory.
ENV MPLCONFIGDIR=/app/cache/matplotlib
ENV YOLO_CONFIG_DIR=/app/cache/ultralytics
ENV HF_HOME=/app/cache/huggingface
ENV TRANSFORMERS_CACHE=/app/cache/huggingface/transformers
ENV TORCH_HOME=/app/cache/torch

RUN mkdir -p $MPLCONFIGDIR $YOLO_CONFIG_DIR $HF_HOME $TRANSFORMERS_CACHE $TORCH_HOME

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Change the ownership of the app directory to the new user.
# This gives the application permission to write to the cache folders.
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Make port 7860 available, as expected by Hugging Face Spaces
EXPOSE 7860

# --- KEY FIX ---
# Command to run the application using Gunicorn with an increased timeout.
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "app:app"]
