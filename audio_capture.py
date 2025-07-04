import time
import queue
import sys
import threading
import os
import wave
import io
import numpy as np
import logging
from collections import deque

# --- Vertex AI Imports ---
import vertexai
from vertexai.generative_models import GenerativeModel, Part

# --- Audio Recording Impoåts ---
import sounddevice as sd

# --- Configuration ---
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms chunks
AUDIO_SEGMENT_DURATION = 5  # seconds
AUDIO_SEGMENT_SAMPLES = SAMPLE_RATE * AUDIO_SEGMENT_DURATION
MAX_CONVERSATION_DURATION = 60  # Maximum conversation duration in seconds
MAX_SEGMENTS = MAX_CONVERSATION_DURATION // AUDIO_SEGMENT_DURATION

# --- Vertex AI Configuration ---
LLM_MODEL_NAME = "gemini-2.0-flash-001"
try:
    GCP_PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
    GCP_LOCATION = os.environ['GOOGLE_CLOUD_LOCATION']
except KeyError as e:
    logging.error(f"Environment variable {e} not set. Please set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.")
    sys.exit(1)

# --- Global Variables & Queues ---
audio_buffer = queue.Queue()
analysis_queue = queue.Queue()
conversation_buffer = deque(maxlen=MAX_SEGMENTS)  # Buffer to store conversation segments
conversation_lock = threading.Lock()
questions_list = [
    "What is your current weight?",
    "What is your age or date of birth?",
    "Do you smoke?",
    "Do you have any allergies?",
    "What is the reason of your visit today?",
    "Are you taking any medications?",
    "How long have you had these symptoms?",
    "Do you have chest pain?",
    "Do you have shortness of breath?",
    "Do you have swollen feet or hands?",
    "Have you recently lost weight?"
]
questions_lock = threading.Lock()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def audio_callback(indata, frames, time_info, status):
    """Callback function for audio recording."""
    if status:
        logging.warning(f"Audio callback status: {status}")
    audio_buffer.put(bytes(indata))

def convert_audio_to_wav(audio_data):
    """Convert raw audio data to WAV format."""
    try:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_data.tobytes())
        wav_buffer.seek(0)
        return wav_buffer.read()
    except Exception as e:
        logging.error(f"Error converting audio to WAV: {e}")
        return None

def accumulate_audio_segment():
    """Accumulate audio chunks until we have a complete segment."""
    accumulated_data = []
    total_samples = 0
    
    while total_samples < AUDIO_SEGMENT_SAMPLES:
        try:
            chunk = audio_buffer.get(timeout=1)  # 1 second timeout
            if chunk is None:  # Stop signal
                return None
                
            # Convert bytes to numpy array
            audio_array = np.frombuffer(chunk, dtype=np.int16)
            accumulated_data.append(audio_array)
            total_samples += len(audio_array)
            
        except queue.Empty:
            if total_samples > 0:  # If we have some data, return what we have
                break
            continue
    
    if not accumulated_data:
        return None
        
    # Concatenate all chunks
    return np.concatenate(accumulated_data)

def analyze_conversation():
    """Analyze the complete conversation using Gemini."""
    try:
        with conversation_lock:
            if not conversation_buffer:
                return None
                
            # Concatenate all segments in the buffer
            complete_audio = np.concatenate(list(conversation_buffer))
            
        # Convert to WAV format
        wav_data = convert_audio_to_wav(complete_audio)
        if wav_data is None:
            return None
            
        # Initialize Vertex AI
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        model = GenerativeModel(LLM_MODEL_NAME)
        
        # Create the prompt with dynamic questions list
        with questions_lock:
            questions_text = "\n".join([f"-{q}" for q in questions_list])
        
        prompt = f"""This is a conversation transcript between a patient and a doctor. 
        The doctor is asking the patient questions to get information about the patient's health.
        When some of the following question are asked, please return a JSON with the follwoing information:
        [        
        {{
            "question": "<question1>",
            "answer": "<answer>"
        }},
        {{
            "question": "<question2>",
            "answer": "<answer2>"
        }},
        ]
        The questions are:
        {questions_text}
        Pick the questions exclusively from this list.
        If during you analysis you cannot fine an answer to the question, return nothing
        If there are multiple answers to the same question, return the most relevant and recent one
        If there are multiple questions, return each question.
        If you cannot find an answer to the question, return the question and "No answer yet in the answer".
        """
        
        # Create the audio part
        audio_part = Part.from_data(wav_data, mime_type="audio/wav")
        
        # Generate content
        response = model.generate_content([prompt, audio_part])
        
        return response.text.strip()
            
    except Exception as e:
        logging.error(f"Error analyzing conversation: {e}")
        return None

def audio_capture_thread():
    """Thread for capturing audio."""
    logging.info("Starting audio capture thread...")
    
    stream = None
    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            channels=1,
            dtype='int16',
            callback=audio_callback
        )
        stream.start()
        logging.info(f"Recording started with sample rate {SAMPLE_RATE} Hz")
        
        while True:
            # Accumulate audio segment
            audio_segment = accumulate_audio_segment()
            if audio_segment is None:
                break
                
            # Add segment to conversation buffer
            with conversation_lock:
                conversation_buffer.append(audio_segment)
            
            # Signal analysis thread to process the updated conversation
            analysis_queue.put(True)
            
    except Exception as e:
        logging.error(f"Error in audio capture thread: {e}")
    finally:
        if stream and stream.active:
            stream.stop()
            stream.close()
        analysis_queue.put(None)  # Signal analysis thread to stop
        logging.info("Audio capture thread stopped")

def analysis_thread():
    """Thread for analyzing audio with Gemini."""
    logging.info("Starting analysis thread...")
    last_analysis_time = 0
    analysis_interval = 5  # Minimum seconds between analyses
    
    while True:
        try:
            # Get signal from queue
            signal = analysis_queue.get()
            if signal is None:  # Stop signal
                break
                
            # Check if enough time has passed since last analysis
            current_time = time.time()
            if current_time - last_analysis_time >= analysis_interval:
                # Analyze the complete conversation
                result = analyze_conversation()
                if result:
                    logging.info(f"Analysis result: {result}")
                last_analysis_time = current_time
                
        except Exception as e:
            logging.error(f"Error in analysis thread: {e}")
            
    logging.info("Analysis thread stopped")

def question_suggestion_thread():
    """Thread for suggesting new questions based on conversation."""
    logging.info("Starting question suggestion thread...")
    last_suggestion_time = 0
    suggestion_interval = 10  # Minimum seconds between suggestions
    
    while True:
        try:
            # Get signal from queue
            signal = analysis_queue.get()
            if signal is None:  # Stop signal
                break
                
            # Check if enough time has passed since last suggestion
            current_time = time.time()
            if current_time - last_suggestion_time >= suggestion_interval:
                with conversation_lock:
                    if not conversation_buffer:
                        continue
                    complete_audio = np.concatenate(list(conversation_buffer))
                
                # Convert to WAV format
                wav_data = convert_audio_to_wav(complete_audio)
                if wav_data is None:
                    continue
                    
                # Initialize Vertex AI
                vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
                model = GenerativeModel(LLM_MODEL_NAME)
                
                # Create the prompt for question suggestion
                prompt = """By listening to the conversation, please suggest what is the most appropriate question that the doctor should ask the patient.
                Return only the question, nothing else."""
                
                # Create the audio part
                audio_part = Part.from_data(wav_data, mime_type="audio/wav")
                
                # Generate content
                response = model.generate_content([prompt, audio_part])
                suggested_question = response.text.strip()
                
                # Add the suggested question to the list if it's not already there
                if suggested_question:
                    with questions_lock:
                        if suggested_question not in questions_list:
                            questions_list.append(suggested_question)
                            print("QUESTIONS LIST: ", questions_list)
                            logging.info(f"New question added: {suggested_question}")
                
                last_suggestion_time = current_time
                
        except Exception as e:
            logging.error(f"Error in question suggestion thread: {e}")
            
    logging.info("Question suggestion thread stopped")

if __name__ == "__main__":
    logging.info("Starting audio analysis application...")
    
    # Start threads
    capture_thread = threading.Thread(target=audio_capture_thread, daemon=True)
    analysis_thread = threading.Thread(target=analysis_thread, daemon=True)
    suggestion_thread = threading.Thread(target=question_suggestion_thread, daemon=True)
    
    capture_thread.start()
    analysis_thread.start()
    suggestion_thread.start()
    
    try:
        # Keep main thread alive
        while capture_thread.is_alive() and analysis_thread.is_alive() and suggestion_thread.is_alive():
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        logging.info("\nCtrl+C received. Stopping...")
        audio_buffer.put(None)  # Signal audio capture to stop
        
    finally:
        # Wait for threads to finish
        capture_thread.join(timeout=5)
        analysis_thread.join(timeout=5)
        suggestion_thread.join(timeout=5)
        logging.info("Application finished.")
