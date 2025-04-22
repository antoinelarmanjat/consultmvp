# audio.py (with Diarization and Vertex AI LLM Question Monitoring - Polling Mode)
import time
import queue
import sys
import threading
import os
from threading import Lock

# --- Vertex AI Imports ---
# These are kept but won't be used in this simplified polling mode
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, HarmCategory

# --- Audio Recording Imports ---
import sounddevice as sd
import numpy as np

# --- Google Cloud Speech Imports ---
from google.cloud import speech

# --- Configuration ---
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)
LANGUAGE_CODE = "en-US"
EXPECTED_SPEAKERS = 2

# --- Vertex AI Configuration ---
# These are kept but won't be used in this simplified polling mode
LLM_MODEL_NAME = "gemini-2.0-flash-001" # Vertex AI model identifier
DIARIZATION_LLM_MODEL_NAME = "gemini-2.5-pro-preview-03-25" # Use a different model for correction if needed
try:
    GCP_PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
    GCP_LOCATION = os.environ['GOOGLE_CLOUD_LOCATION']
except KeyError as e:
    print(f"ERROR: Environment variable {e} not set.")
    print("Please set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.")
    sys.exit(1)

# --- Global Variables & Queues ---
audio_buffer = queue.Queue()
# processing_queue is removed as run_transcription now updates transcript_segments directly
# emission_queue is removed as we are not using SocketIO push
questions_list = []  # Global questions list (kept for initial state route)
questions_lock = threading.Lock()  # Lock for thread-safe access to questions_list
questions_loaded = False  # Flag to track if questions have been loaded
conversation_history = [] # Used by LLM processing (kept but won't grow in this mode)
conversation_lock = Lock() # New: Also store segments for initial client load, updated by process_transcripts
transcript_segments = [] # This list will now be updated by run_transcription
transcript_lock = Lock() # Lock for thread-safe access to transcript_segments

diarization_correction_history = [] # Store corrected versions (kept but won't grow)
diarization_correction_lock = Lock()
segment_count = 0 # Count segments processed (kept but won't grow)
SEGMENTS_BEFORE_CORRECTION = 7 # (kept but won't be used)

# We no longer pass socketio_instance to threads in this polling mode

def get_current_questions():
    """Returns a copy of the current questions list."""
    with questions_lock:
        return [q.copy() for q in questions_list]

def get_transcript_segments():
     """Returns a copy of the current transcript segments."""
     with transcript_lock: # Use the lock when accessing transcript_segments
         print(f"[GETTER] Getting {len(transcript_segments)} segments from transcript_segments.") # Added log
         return transcript_segments.copy()

def get_diarization_correction_history():
     """Returns a copy of the diarization correction history."""
     with diarization_correction_lock:
         return diarization_correction_history.copy()


# --- Helper Function to Load Questions (Unchanged) ---
def load_questions(filename="questions.txt"):
    """Loads questions from a file into a list of dictionaries."""
    global questions_list, questions_loaded
    with questions_lock:
        if not questions_loaded:
            questions_list = []
            try:
                with open(filename, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            questions_list.append({
                                'text': line,
                                'status': 'pending', # 'pending', 'suggested', 'asked', 'answered'
                                'answer': None,
                                'general': len(questions_list) < 3  # First 3 questions are general
                            })
                questions_loaded = True
                print(f"Loaded {len(questions_list)} questions from {filename}")
                print("Loaded questions:")
                for q in questions_list:
                    print(f"  - {q['text']} (status: {q['status']}, general: {q['general']})")
                return questions_list
            except FileNotFoundError:
                print(f"ERROR: {filename} not found. Please create it.")
                return []
        else:
            print("Questions already loaded, returning existing list")
            return questions_list


# --- Audio Recording Callback (Unchanged) ---
def audio_callback(indata, frames, time_info, status):
    if status:
        # Handle status if needed, e.g., stream overflow
        pass # For now, just ignore status
    audio_buffer.put(bytes(indata))

# --- Generator for Audio Stream to API (Unchanged) ---
def audio_generator():
    while True:
        chunk = audio_buffer.get()
        if chunk is None:
            break
        yield speech.StreamingRecognizeRequest(audio_content=chunk)

# --- Speech-to-Text Thread Function (MODIFIED to update transcript_segments directly) ---
# No longer uses processing_queue or emits SocketIO events
def run_transcription(): # Removed processing_queue and socketio_instance arguments
    client = speech.SpeechClient()

    # --- Configuration ---
    # Diarization config is kept, but diarization results won't be processed by LLM in this mode
    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=True,
        min_speaker_count=EXPECTED_SPEAKERS,
        max_speaker_count=EXPECTED_SPEAKERS,
    )
    recognition_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=LANGUAGE_CODE,
        enable_automatic_punctuation=True,
        diarization_config=diarization_config,
        enable_word_confidence=True, # Keep this, it helps inspect word details
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=recognition_config,
        interim_results=True,
    )
    print("\n[Transcription Thread] Starting Google Cloud Speech-to-Text stream...")
    requests = audio_generator()
    stream_ended_normally = False # Flag to check if loop finishes

    try:
        responses = client.streaming_recognize(
            config=streaming_config,
            requests=requests,
        )
        print("[Transcription Thread] Receiving responses from Google API...")

        # --- Log each response received ---
        for i, response in enumerate(responses):
            sys.stdout.flush()

            if not response.results:
                print("-> No results in response.")
                continue

            result = response.results[0]
            sys.stdout.flush()

            if not result.alternatives:
                print("-> No alternatives in result.")
                continue

            # Always print interim results for user feedback
            if not result.is_final:
                interim_transcript = result.alternatives[0].transcript
                print(f"Interim: {interim_transcript}", end='\r') # Use \r to overwrite line
                sys.stdout.flush()
                continue # Go to the next response

            # --- This block is ONLY reached if result.is_final is True ---
            print(f"\n[Transcription Thread] Final Segment {i+1} received.")
            sys.stdout.flush()

            final_alternative = result.alternatives[0]
            if not final_alternative.words:
                print("[Transcription Thread] Final result has no words.")
                continue

            # --- Diarization/Formatting (Only if diarization is enabled) ---
            if hasattr(recognition_config, 'diarization_config') and recognition_config.diarization_config.enable_speaker_diarization:
                 diarized_segment = ""
                 current_speaker_tag = None
                 for word_info in final_alternative.words:
                     tag = word_info.speaker_tag if hasattr(word_info, 'speaker_tag') else 0
                     word_text = word_info.word

                     if current_speaker_tag is None or tag != current_speaker_tag:
                         if current_speaker_tag is not None:
                            diarized_segment += "\n"
                         current_speaker_tag = tag
                         diarized_segment += f"[Speaker {current_speaker_tag}]: {word_text}"
                     else:
                         diarized_segment += f" {word_text}"
            else:
                 diarized_segment = final_alternative.transcript
                 print("[Transcription Thread] Diarization disabled, using raw transcript.")

            print(f"\n[Transcription Thread] --- FINAL DIARIZED SEGMENT ----\n{diarized_segment}\n------------------------------------")

            # --- Append the final segment to the global transcript_segments list ---
            with transcript_lock: # Use the lock when modifying the shared list
                transcript_segments.append(diarized_segment)
                print(f"[Transcription Thread] Appended segment to transcript_segments. Total segments: {len(transcript_segments)}")


        # --- End of the 'for response in responses:' loop ---
        stream_ended_normally = True
        print("\n[Transcription Thread] 'responses' iterator finished.")

    except Exception as e:
        print(f"\n[Transcription Thread] !!! FATAL ERROR during transcription stream: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

    finally:
        print(f"\n[Transcription Thread] Transcription stream thread stopping. Stream ended normally: {stream_ended_normally}.")
        # No queues to signal stop to in this simplified mode


# --- process_transcripts thread function is removed in this polling mode ---
# --- diarization_correction_thread function is removed in this polling mode ---
# --- socketio_emitter_thread function is removed in this polling mode ---
# --- emission_queue is removed ---
# --- handle_diarization_correction is removed ---


# --- Main Execution Block (Adjusted for passing socketio if run standalone) ---
# Note: This __main__ block is typically not used when run via app.py,
# but kept for completeness if you wanted to test audio.py directly (without Flask/SocketIO)
if __name__ == "__main__":
    print("Starting audio processing application with Diarization and Vertex AI LLM Monitoring (Standalone Polling Mode)...")
    print("Note: This mode only transcribes and stores segments. It does not serve a web page or perform LLM analysis.")

    # --- Start Threads ---
    # We only need the transcription thread
    transcription_thread = threading.Thread(target=run_transcription, daemon=True)

    transcription_thread.start()

    # --- Start Recording Audio ---
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
        print("Recording started. Press Ctrl+C to stop.")

        # Keep main thread alive while the transcription thread is alive
        while transcription_thread.is_alive():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping recording...")
        audio_buffer.put(None) # Signal audio generator to stop

    except Exception as e:
        print(f"\nAn unexpected error occurred in the main loop: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Ensure thread is signaled to stop even on other errors
        if audio_buffer.empty(): audio_buffer.put(None)


    finally:
        # Cleanup
        if stream and stream.active:
             print("Stopping audio stream...")
             stream.stop()
             stream.close()

        print("Waiting for threads to finish...")
        # transcription_thread.join(timeout=5) # Join the transcription thread


        print("Application finishing.")
