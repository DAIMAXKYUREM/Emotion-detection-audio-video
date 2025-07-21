# text_model.py

import torch
import os

# Check for GPU availability to decide which model loading strategy to use
IS_GPU_AVAILABLE = torch.cuda.is_available()

if IS_GPU_AVAILABLE:
    from unsloth import FastLanguageModel
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer

class TextSummarizer:
    """
    A class to generate a summary or analysis of text using a Llama 3 model,
    factoring in emotional context from face and voice.
    It dynamically loads the model using Unsloth for GPU or standard Transformers for CPU.
    """
    # --- KEY FIX ---
    # Corrected the model identifier for the CPU version.
    def __init__(self, gpu_model="unsloth/llama-3-8b-Instruct-bnb-4bit", cpu_model="meta-llama/Meta-Llama-3-8B-Instruct"):
        """
        Initializes the TextSummarizer, loading the Llama 3 model.
        """
        self.device = "cuda" if IS_GPU_AVAILABLE else "cpu"
        print(f"Initializing TextSummarizer on device: {self.device}")

        if IS_GPU_AVAILABLE:
            self._init_gpu_model(gpu_model)
        else:
            self._init_cpu_model(cpu_model)

    def _init_gpu_model(self, model_name):
        """Loads the model using Unsloth for GPU acceleration."""
        print(f"Loading GPU-optimized model: {model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        print("GPU model loaded successfully.")

    def _init_cpu_model(self, model_name):
        """Loads the model using standard Transformers for CPU execution."""
        print(f"Loading CPU-compatible model: {model_name}")
        # For CPU, we don't use 4-bit loading.
        # We specify torch_dtype to use bfloat16 if available, for better performance.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("CPU model loaded successfully.")


    def summarize_with_context(self, text: str, facial_emotion: str, voice_emotion: str) -> str:
        """
        Analyzes the text in the context of detected emotions.

        Args:
            text (str): The transcribed text from the user's speech.
            facial_emotion (str): The emotion detected from the user's face.
            voice_emotion (str): The emotion detected from the user's voice.

        Returns:
            str: A concise analysis or summary of the input.
        """
        if not text:
            return "No text provided to analyze."

        # Using the Llama 3 Instruct prompt format
        messages = [
            {
                "role": "system",
                "content": "You are an expert analyst. Your task is to provide a concise, one-sentence summary of the user's statement. You must consider the emotional context provided by their facial expression and tone of voice.",
            },
            {
                "role": "user", 
                "content": f"""Analyze the following statement and provide a summary.
- Statement: "{text}"
- Facial Emotion Detected: {facial_emotion}
- Vocal Emotion Detected: {voice_emotion}
"""
            },
        ]
        
        # Prepare the input for the model
        # The tokenizer for both models works the same way.
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate the response
        # The generate method is also standard.
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=50,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode the response
        response = self.tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        return response.strip()

if __name__ == '__main__':
    # Example usage
    try:
        summarizer = TextSummarizer()
        
        test_text = "I can't believe it's already Friday, this week just flew by."
        test_face_emotion = "surprise"
        test_voice_emotion = "HAPPY"
        
        print(f"\n--- Analyzing Text ---")
        print(f"Text: '{test_text}'")
        print(f"Facial Emotion: {test_face_emotion}")
        print(f"Vocal Emotion: {test_voice_emotion}")
        
        analysis = summarizer.summarize_with_context(test_text, test_face_emotion, test_voice_emotion)
        
        print("\n--- Llama 3 Analysis ---")
        print(analysis)
        
    except Exception as e:
        print(f"An error occurred during standalone test: {e}")