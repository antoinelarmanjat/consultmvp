# audio.py (with Speaker Diarization)
import time
import queue
import sys
import threading

# --- Audio Recording Imports ---
import sounddevice as sd
import numpy as np

# --- Google Cloud Speech Imports ---
from google.cloud import speech

# --- Configuration ---
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)
LANGUAGE_CODE = "en-US" # Adjust if needed (e.g., "fr-FR")
EXPECTED_SPEAKERS = 1 # For Patient + Doctor conversation

# --- Global Variables & Queues ---
audio_buffer = queue.Queue()
processing_queue = queue.Queue()
# full_transcript_parts = [] # Still optional for history
# transcript_lock = threading.Lock() # Still optional

# --- Audio Recording Callback ---
def audio_callback(indata, frames, time_info, status):
    if status:
        print(status, file=sys.stderr)
    audio_buffer.put(bytes(indata))

# --- Generator for Audio Stream to API ---
def audio_generator():
    while True:
        chunk = audio_buffer.get()
        if chunk is None:
            break
        yield speech.StreamingRecognizeRequest(audio_content=chunk)

# --- Speech-to-Text Thread Function ---
def run_transcription():
    # global full_transcript_parts # Uncomment if using the history list

    client = speech.SpeechClient()

    # --- *** DIARIZATION CONFIGURATION *** ---
    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization=False,
        min_speaker_count=EXPECTED_SPEAKERS, # Minimum number of speakers expected
        max_speaker_count=EXPECTED_SPEAKERS, # Maximum number of speakers expected (can be > min)
        # Speaker tags start at 1
    )
    # ---------------------------------------

    recognition_config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code=LANGUAGE_CODE,
        enable_automatic_punctuation=True,
        # --- *** ADD DIARIZATION CONFIG *** ---
        diarization_config=diarization_config,
        # ------------------------------------
        # ------------------------------------
        # model="medical_conversation", # Consider specialized models
        # use_enhanced=True,
        # --- Enable word-level confidence and speaker tags ---
        # These are often enabled implicitly by diarization, but explicit is safe
        enable_word_confidence=True, # Not strictly needed for tags, but good practice
        # Diarization implicitly requires word-level details
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=recognition_config,
        interim_results=True,
    )

    print("Starting Google Cloud Speech-to-Text stream with Diarization...")
    requests = audio_generator()

    try:
        responses = client.streaming_recognize(
            config=streaming_config,
            requests=requests,
        )

        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            # Process interim results (without speaker tags usually)
            if not result.is_final:
                transcript_chunk = result.alternatives[0].transcript
                print(f"Interim: {transcript_chunk}", end='\r')
                continue # Skip to next response for interim

            # --- *** PROCESS FINAL RESULT WITH DIARIZATION *** ---
            final_alternative = result.alternatives[0]
            if not final_alternative.words:
                print("Warning: Final result received without word info, skipping diarization for this segment.")
                continue

            diarized_segment = ""
            current_speaker_tag = None # Track the speaker tag for the current word sequence

            for word_info in final_alternative.words:
                # word_info contains: word, start_time, end_time, confidence, speaker_tag
                if current_speaker_tag is None or word_info.speaker_tag != current_speaker_tag:
                    # Speaker change detected or first word
                    if current_speaker_tag is not None:
                         diarized_segment += "\n" # Add newline between speaker turns for clarity
                    current_speaker_tag = word_info.speaker_tag
                    diarized_segment += f"[Speaker {current_speaker_tag}]: {word_info.word}"
                else:
                    # Same speaker, just append the word
                    diarized_segment += f" {word_info.word}"

            print(f"Final Segment:\n{diarized_segment}") # Console feedback

            # Put the fully constructed diarized segment onto the processing queue
            processing_queue.put(diarized_segment)

            # --- Optional: Store in historical list (if needed) ---
            # with transcript_lock:
            #     full_transcript_parts.append(diarized_segment)
            # ------------------------------------------------------

            # --- *** END OF DIARIZATION PROCESSING *** ---

    except Exception as e:
        print(f"Error during transcription stream: {e}")
    finally:
        print("Transcription stream stopped.")
        processing_queue.put(None) # Signal processing thread to stop


# --- Transcript Processing Thread Function ---
def process_transcripts():
    """Runs in a separate thread, processing DIARIZED transcript segments."""
    print("Processing thread started, waiting for diarized segments...")
    while True:
        segment = processing_queue.get() # Blocks until an item is available

        if segment is None:
            print("Processing thread received stop signal.")
            break

        # --- Perform your actions on the DIARIZED 'segment' here ---
        # The 'segment' variable now contains text like:
        # "[Speaker 1]: Hello Doctor Smith.\n[Speaker 2]: Hello patient, how are you feeling today?"

        print(f"[Processor] Received Segment:\n{segment}") # Example action

        # Example Action: Separate segments by speaker for analysis
        # speaker_texts = {}
        # try:
        #     for line in segment.strip().split('\n'):
        #         if line.startswith("[Speaker"):
        #             parts = line.split("]: ", 1)
        #             if len(parts) == 2:
        #                 speaker_tag = parts[0].split(" ")[1] # Extract tag number
        #                 text = parts[1]
        #                 if speaker_tag not in speaker_texts:
        #                     speaker_texts[speaker_tag] = []
        #                 speaker_texts[speaker_tag].append(text)
        #     print(f"[Processor] Parsed Speaker Texts: {speaker_texts}")

            # Now you could, for example, send only Speaker 1's text to one Gemini prompt
            # and Speaker 2's text to another, or analyze turn-taking, etc.

        # except Exception as e:
        #     print(f"[Processor] Error parsing diarized segment: {e}")

        # Add your other specific actions here, operating on the diarized segment.
        # ----------------------------------------------------

    print("Processing thread finished.")


# --- Main Execution Block (Largely unchanged, ensure imports/config at top are updated) ---
if __name__ == "__main__":
    # --- Prerequisites Reminder ---
    # 1. Google Cloud Project with Speech-to-Text API enabled.
    # 2. `pip install google-cloud-speech sounddevice numpy`
    # 3. Authenticated via `gcloud auth application-default login`
    # 4. Diarization might have language/model limitations - check docs if issues arise.
    # --------------------------------

    print("Starting audio processing application with Speaker Diarization...")
    try:
        print(f"Using input device: {sd.query_devices(kind='input')['name']}")
    except Exception as e:
        print(f"Could not query audio devices: {e}. Using default.")

    print(f"Sample rate: {SAMPLE_RATE}, Chunk size: {CHUNK_SIZE}, Language: {LANGUAGE_CODE}, Speakers: {EXPECTED_SPEAKERS}")

    # --- Start Threads ---
    transcription_thread = threading.Thread(target=run_transcription, daemon=True)
    processing_thread = threading.Thread(target=process_transcripts, daemon=True)

    transcription_thread.start()
    processing_thread.start()

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

        while transcription_thread.is_alive() and processing_thread.is_alive():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping recording and processing...")
        audio_buffer.put(None) # Signal audio generator

    except Exception as e:
        print(f"\nAn unexpected error occurred in the main loop: {e}")
        if audio_buffer.empty():
             audio_buffer.put(None)

    finally:
        # --- Cleanup ---
        if stream and stream.active:
             print("Stopping audio stream...")
             stream.stop()
             stream.close()

        print("Waiting for threads to finish...")
        if transcription_thread.is_alive():
            transcription_thread.join(timeout=5)
        if processing_thread.is_alive():
            processing_thread.join(timeout=5)

        print("All threads finished.")
        print("Application finished.")

