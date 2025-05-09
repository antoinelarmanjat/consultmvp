# audio.py (with Diarization and Vertex AI LLM Question Monitoring)
import time
import queue
import sys
import threading
import os
from threading import Lock
import sqlite3
from datetime import datetime
import logging # Added logging

# --- Vertex AI Imports ---
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting, HarmCategory

# --- Audio Recording Imports ---
import sounddevice as sd
import numpy as np

# --- Google Cloud Speech Imports ---
from google.cloud import speech

# --- Configuration ---
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10) # 100ms chunks
LANGUAGE_CODE = "en-US"
EXPECTED_SPEAKERS = 2 # For diarization

# --- Vertex AI Configuration ---
LLM_MODEL_NAME = "gemini-2.0-flash-001"
DIARIZATION_LLM_MODEL_NAME = "gemini-2.5-pro-preview-03-25"
try:
    GCP_PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']
    GCP_LOCATION = os.environ['GOOGLE_CLOUD_LOCATION']
except KeyError as e:
    logging.error(f"Environment variable {e} not set. Please set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION.")
    sys.exit(1)

# --- Global Variables & Queues ---
audio_buffer = queue.Queue()
processing_queue = queue.Queue() # For final transcripts to be processed by process_transcripts
questions_list = []
questions_lock = threading.Lock()
questions_loaded = False
conversation_history = []
conversation_lock = Lock()
transcript_segments = []
transcript_lock = Lock()

diarization_correction_history = []
diarization_correction_lock = Lock()
segment_count = 0
SEGMENTS_BEFORE_CORRECTION = 4

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Output logs to stdout
    ]
)

def get_current_questions():
    """Returns a copy of the current questions list."""
    with questions_lock:
        return [q.copy() for q in questions_list]

def get_transcript_segments():
     """Returns a copy of the current transcript segments."""
     with transcript_lock:
         return transcript_segments.copy()

def get_diarization_correction_history():
     """Returns a copy of the diarization correction history."""
     with diarization_correction_lock:
         return diarization_correction_history.copy()

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
                                'status': 'pending',
                                'answer': None,
                                'general': len(questions_list) < 6
                            })
                questions_loaded = True
                logging.info(f"Loaded {len(questions_list)} questions from {filename}")
                for q in questions_list:
                    logging.debug(f"  - {q['text']} (status: {q['status']}, general: {q['general']})")
                return questions_list
            except FileNotFoundError:
                logging.error(f"{filename} not found. Please create it.")
                return []
        else:
            logging.info("Questions already loaded, returning existing list")
            return questions_list

# --- Audio Recording Callback (MODIFIED for better status reporting) ---
def audio_callback(indata, frames, time_info, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        # More descriptive logging for audio issues
        logging.warning(f"[Audio Callback] Status flags: {status} - {sd.CallbackFlags(status)}")
        if status & sd.CallbackFlags.INPUT_UNDERFLOW:
            logging.warning("[Audio Callback] Input underflow: not enough data from the audio interface.")
        if status & sd.CallbackFlags.INPUT_OVERFLOW:
            logging.warning("[Audio Callback] Input overflow: data from the audio interface was lost.")
        # You might add more specific handling or counters for these issues if needed
    audio_buffer.put(bytes(indata))

# --- Generator for Audio Stream to API (Unchanged) ---
def audio_generator():
    """Yields audio chunks from the buffer to the Speech API.
    This is a generator function that blocks until data is available.
    """
    logging.info("[Audio Generator] Starting...")
    while True:
        chunk = audio_buffer.get() # Blocks here until audio data is available
        if chunk is None:
            logging.info("[Audio Generator] Received None, stopping.")
            break
        yield speech.StreamingRecognizeRequest(audio_content=chunk)
    logging.info("[Audio Generator] Stopped.")


# --- Speech-to-Text Thread Function (IMPROVED) ---
def run_transcription(processing_queue_ref): # Renamed arg to avoid conflict with global
    """
    Continuously captures audio, sends it to Google Cloud Speech-to-Text,
    formats the final diarized transcript, and puts it on the processing_queue.
    """
    client = speech.SpeechClient()

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
        enable_word_confidence=True, # Useful for debugging, little overhead
        # model="telephony", # Example: consider if your audio is phone-like
        # use_enhanced=True, # For certain models
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=recognition_config,
        interim_results=True,
    )

    logging.info("Starting Google Cloud Speech-to-Text stream...")
    audio_requests = audio_generator() # Get the generator
    stream_ended_normally = False
    last_interim_output_time = time.time()

    try:
        responses = client.streaming_recognize(
            config=streaming_config,
            requests=audio_requests,
        )

        logging.info("Receiving responses from Google API...")
        for response_idx, response in enumerate(responses):
            if response.error and response.error.message:
                logging.error(f"Google Speech API Error: {response.error.message} (Code: {response.error.code})")
                # Depending on the error code, you might want to break or attempt to restart.
                # For simplicity, we'll break here.
                # See https://cloud.google.com/apis/design/errors#error_codes for codes
                if response.error.code == 11: # DEADLINE_EXCEEDED for streaming
                     logging.warning("Deadline exceeded, may indicate silence or network issues. Stream might close.")
                # Some errors might be terminal for the stream.
                break # Stop processing this stream

            if not response.results:
                logging.debug(f"Response {response_idx}: No results.")
                continue

            result = response.results[0]
            if not result.alternatives:
                logging.debug(f"Response {response_idx}: No alternatives in result.")
                continue

            transcript_alternative = result.alternatives[0]

            if result.is_final:
                # Clear any previous interim line
                sys.stdout.write("\r" + " " * 80 + "\r") # Clear line
                sys.stdout.flush()
                logging.info(f"Final Segment {response_idx + 1} received (Stability: {result.stability:.2f}).")

                if not transcript_alternative.words:
                    logging.warning("Final result has no words, skipping.")
                    continue

                diarized_segment = ""
                current_speaker_tag = -1 # Initialize to a value that won't match first tag

                if diarization_config.enable_speaker_diarization:
                    for i, word_info in enumerate(transcript_alternative.words):
                        # Google Speech API speaker_tag is usually an integer (1, 2, ...).
                        # Tag 0 can sometimes mean "unknown" or "undetermined".
                        speaker_tag_str = "UNKNOWN" # Default for missing or 0 tag
                        if hasattr(word_info, 'speaker_tag') and word_info.speaker_tag != 0:
                            speaker_tag_str = str(word_info.speaker_tag)
                        elif hasattr(word_info, 'speaker_tag') and word_info.speaker_tag == 0 :
                            # Explicitly handle tag 0 if it means something specific or keep as UNKNOWN
                             speaker_tag_str = "UNKNOWN" # Or "SPEAKER_0" if you prefer

                        word_text = word_info.word

                        # Start of new speaker segment or first word
                        if current_speaker_tag != speaker_tag_str:
                            if diarized_segment: # Add newline if not the very first part of the segment
                                diarized_segment += "\n"
                            current_speaker_tag = speaker_tag_str
                            diarized_segment += f"[Speaker {current_speaker_tag}]: {word_text}"
                        else:
                            diarized_segment += f" {word_text}"
                else:
                    diarized_segment = transcript_alternative.transcript
                    logging.debug("Diarization disabled, using raw transcript for segment.")

                logging.info(f"--- FINAL DIARIZED SEGMENT ---\n{diarized_segment}\n------------------------------------")

                # Put the complete, final, diarized segment onto the queue for further processing
                if diarized_segment: # Ensure we don't put empty strings
                    logging.debug(f"Attempting to put final segment into processing queue...")
                    processing_queue_ref.put(diarized_segment)
                    logging.debug("Final segment put in processing queue.")
                else:
                    logging.warning("Empty diarized segment was not added to processing queue.")

            else: # Interim result
                interim_transcript = transcript_alternative.transcript
                # Limit interim output frequency to avoid flooding logs/console
                current_time = time.time()
                if current_time - last_interim_output_time > 0.5: # Log interim max every 0.5s
                    # Use \r to overwrite the same line in the console for a live feel
                    sys.stdout.write(f"\rInterim: {interim_transcript[:100]}...")
                    sys.stdout.flush()
                    last_interim_output_time = current_time
                logging.debug(f"Interim transcript: {interim_transcript}")


        # End of the 'for response in responses:' loop
        # This is reached if the 'responses' iterator finishes (e.g., audio_generator yielded None)
        stream_ended_normally = True
        logging.info("'responses' iterator finished normally.")

    except StopIteration:
        # This can happen if the audio_generator() stops yielding (e.g. audio_buffer.put(None) was called)
        stream_ended_normally = True
        logging.info("Transcription stream stopped due to StopIteration (audio_generator likely finished).")
    except Exception as e:
        logging.error(f"FATAL ERROR during transcription stream: {type(e).__name__}: {e}", exc_info=True)
        # exc_info=True will print the full traceback

    finally:
        # Clear any lingering interim line
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()
        logging.info(f"Transcription stream thread stopping. Stream ended normally: {stream_ended_normally}.")
        # Signal the processing thread that transcription is done by putting None in its queue.
        logging.info("Signaling processing queue to stop.")
        processing_queue_ref.put(None)


# --- Transcript Processing Thread Function (Unchanged as per request) ---
def process_transcripts(processing_queue_ref, patient_id):
    """Runs in a separate thread, processing segments, managing questions, and updating history."""
    global segment_count, conversation_history, transcript_segments

    logging.info("Processing thread started, loading questions...")
    load_questions()
    logging.debug(f"Initial QUESTIONS LIST in process_transcripts: {questions_list}")

    try:
        logging.info(f"Initializing Vertex AI for Project '{GCP_PROJECT_ID}' in Location '{GCP_LOCATION}'...")
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI: {e}", exc_info=True)
        return

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    try:
        model = GenerativeModel(LLM_MODEL_NAME)
        logging.info(f"Vertex AI model '{LLM_MODEL_NAME}' loaded for questions.")
    except Exception as e:
        logging.error(f"Could not load Vertex AI model '{LLM_MODEL_NAME}' for questions: {e}", exc_info=True)
        return

    logging.info("\n" + "="*15 + " Initial General Questions " + "="*15)
    patient_info = get_patient_info(patient_id)
    
    if patient_info:
        logging.info(f"Found patient info: {patient_info}")
        db_info_prompt = f"""
        Here is the patient information from the database:
        {patient_info}
        
        Here is the list of questions that need to be asked:
        {[q['text'] for q in questions_list if q['general'] and q['status'] == 'pending']}
        
        For each question, determine if the information is already available in the database.
        If yes, provide the answer from the database.
        If no, mark it as needing to be asked.
        
        If the question is about age and you have the birthdate, use the calculated age.
        For smoking status, use "Yes" for smokers and "No" for non-smokers.
        
        Respond with a JSON array of objects, where each object has:
        - question: the exact question text from the list
        - answer: either the answer from the database or "needs to be asked"
        
        Example format:
        [
            {{"question": "Do you smoke?", "answer": "No"}},
            {{"question": "What is your age?", "answer": "26"}},
            {{"question": "What are your allergies?", "answer": "needs to be asked"}}
        ]
        
        Only include questions from the provided list. Do not add any other text or explanation.
        """
        try:
            response = model.generate_content(db_info_prompt, safety_settings=safety_settings)
            analysis = response.text.strip()
            logging.debug(f"LLM DB Analysis Raw: {analysis}")
            
            if analysis.startswith('```json'):
                analysis = analysis[7:]
            if analysis.endswith('```'):
                analysis = analysis[:-3]
            analysis = analysis.strip()
            
            import json
            try:
                answers = json.loads(analysis)
                for answer_obj in answers: # Renamed to avoid conflict
                    question_text = answer_obj['question']
                    answer_text = answer_obj['answer']
                    with questions_lock:
                        for question in questions_list:
                            if question['text'] == question_text and question['status'] == 'pending':
                                if answer_text != "needs to be asked":
                                    question['status'] = 'answered'
                                    question['answer'] = answer_text
                                    logging.info(f"Question already answered from database: {question_text} -> Answer: {answer_text}")
                                else:
                                    question['status'] = 'suggested'
                                    logging.info(f"Question needs to be asked (from DB check): {question_text}")
                                break
            except json.JSONDecodeError as e:
                logging.error(f"Error parsing LLM DB analysis JSON: {e}. Raw response: {analysis}", exc_info=True)
                with questions_lock:
                    for question in questions_list:
                        if question['general'] and question['status'] == 'pending':
                            question['status'] = 'suggested'
        except Exception as e:
            logging.error(f"Error analyzing database info with LLM: {e}", exc_info=True)
            with questions_lock:
                for question in questions_list:
                    if question['general'] and question['status'] == 'pending':
                        question['status'] = 'suggested'
    else:
        logging.info("No patient info found in database, suggesting all general questions.")
        with questions_lock:
            for question in questions_list:
                if question['general'] and question['status'] == 'pending':
                    question['status'] = 'suggested'
    
    logging.info("="*51 + "\n")
    initial_suggestions_done = True

    while True:
        segment = processing_queue_ref.get() # Use the reference

        if segment is None:
            logging.info("Processing thread received stop signal.")
            break

        logging.info(f"\n[Processor] Processing Segment:\n{segment}\n-----------------------------")

        with conversation_lock:
            conversation_history.append(segment)
            segment_count += 1
            logging.debug(f"Segment count updated to: {segment_count}")


        with transcript_lock:
            transcript_segments.append(segment)

        updated_questions_during_segment_processing = False
        with questions_lock:
            #questions_to_check = [q for q in questions_list if q['status'] == 'suggested']
            questions_to_check = [q for q in questions_list if q['status'] == 'suggested']
        
        logging.debug(f"Questions to check for answers: {[q['text'] for q in questions_to_check]}")

        for question in questions_to_check: # Iterate on a copy or re-fetch if modification happens inside
            full_context = "\n".join(conversation_history) # Use current conversation history
            prompt = f"""
            This is a conversation transcript of patient and a doctor. The conversation is difficult to follow
            because the diarization is not very good. The goal is to extract the answer to questions that
            are asked by the doctor during the consultation, but that are difficult to find in the conversation.
            Here is the conversation transcript:
            ---
            {full_context}
            ---
            Has the question "{question['text']}" been answered or addressed in the MOST RECENT part of the conversation, or earlier?
            Dont pay any attention to the diarization because it does not work, try to figure out the answers without
            taking care of the Speaker indications. For symptoms, consider what is said in the broader sense, don't try and match
            the question too literally.
            Respond in one of two ways EXACTLY:
            1. If YES (or likely), respond with: YES - [The specific answer extracted from the conversation]
            2. If NO, respond with: NO
            """
            try:
                response = model.generate_content(prompt, safety_settings=safety_settings)
                llm_answer = response.text.strip()
                logging.debug(f"LLM Answer check for Q '{question['text']}': {llm_answer}")

                if llm_answer.startswith("YES -"):
                    extracted_answer = llm_answer[len("YES -"):].strip()
                    with questions_lock: # Lock before modifying shared questions_list
                        # Re-find the question in the main list to ensure its current state
                        for q_main in questions_list:
                            if q_main['text'] == question['text']:
                                if q_main['status'] != 'answered' or q_main['answer'] != extracted_answer:
                                    q_main['status'] = 'answered'
                                    q_main['answer'] = extracted_answer
                                    updated_questions_during_segment_processing = True
                                    logging.info("="*40)
                                    logging.info(f"[System Found Answer]")
                                    logging.info(f"  Q: {q_main['text']}")
                                    logging.info(f"  A: {extracted_answer}")
                                    logging.info("="*40 + "\n")
                                break
            except Exception as e:
                logging.error(f"[Processor] ERROR during Vertex AI API call for answer check '{question['text']}': {e}", exc_info=True)

        updated_suggestions_during_segment_processing = False
        if initial_suggestions_done:
            with questions_lock:
                pending_contextual_questions = [
                    q for q in questions_list if not q['general'] and q['status'] == 'pending'
                ]
            
            if pending_contextual_questions:
                full_context = "\n".join(conversation_history)
                candidate_question_texts = [q['text'] for q in pending_contextual_questions]
                suggestion_prompt = f"""You are an assistant helping a doctor during a patient consultation.
                Analyze the following ongoing conversation transcript:
                --- CONVERSATION START ---
                {full_context}
                --- CONVERSATION END ---

                Here is a list of potential follow-up questions that have NOT YET been asked or answered:
                --- AVAILABLE QUESTIONS ---
                {chr(10).join(f'- {q}' for q in candidate_question_texts)}
                --- END AVAILABLE QUESTIONS ---
                Don't take into account the diarization because it might be completely off and incorrect.
                Try to find the answer without the speaker indications (maybe wrong speakers)
                Based on the flow and content of the conversation, identify ALL questions from the list above that have become relevant and appropriate for the doctor to ask NEXT.

                Consider:
                1. The patient's current symptoms or complaints
                2. Any medical history mentioned
                3. The natural flow of a medical consultation
                4. Questions that would help clarify the patient's condition

                List EACH relevant question's exact text on a new line.
                If NONE of the available questions seem particularly relevant to ask right now based on the latest developments, respond ONLY with the word "NONE". Do not add any other text or explanation.
                """
                try:
                    response = model.generate_content(suggestion_prompt, safety_settings=safety_settings)
                    suggested_lines = response.text.strip().splitlines()
                    logging.debug(f"LLM Suggested contextual questions: {suggested_lines}")

                    if suggested_lines and suggested_lines[0].strip().lower() != "none":
                        logging.info("\n[System Suggests Asking (Contextual)]:")
                        for suggested_text in suggested_lines:
                            suggested_text = suggested_text.strip()
                            if not suggested_text: continue
                            with questions_lock: # Lock before modifying shared questions_list
                                for question_obj in pending_contextual_questions: # Iterate over the local copy for comparison
                                     # Re-find in main list to update actual object
                                    for q_main in questions_list:
                                        if q_main['text'] == question_obj['text']:
                                            if (q_main['text'].strip().lower() == suggested_text.lower() or \
                                                suggested_text.lower() in q_main['text'].strip().lower()) and \
                                               q_main['status'] == 'pending': # Double check status
                                                logging.info(f"  - {q_main['text']}")
                                                q_main['status'] = 'suggested'
                                                updated_suggestions_during_segment_processing = True
                                            break # Found the question in main list
                                    
                except Exception as e:
                    logging.error(f"[Processor] ERROR during Vertex AI API call for question suggestion: {e}", exc_info=True)

    logging.info("\n--- Final Question Status ---")
    with questions_lock:
        for q in questions_list:
            logging.info(f"Q: {q['text']}")
            logging.info(f"  Status: {q['status']}")
            if q['answer']:
                logging.info(f"  Answer: {q['answer']}")
    logging.info("---------------------------\n")
    logging.info("Processing thread finished.")


# --- Diarization Correction Helper (Unchanged) ---
def send_to_gemini_for_diarization(conversation_text):
    """Send conversation to Gemini for diarization correction."""
    try:
        # Consider initializing model once if this function is called frequently by the same thread.
        # For now, keeping original behavior.
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION) #Potentially re-init
        model = GenerativeModel(DIARIZATION_LLM_MODEL_NAME)
        prompt = f"""
        The diarization of Google speech to text does not work very well. Here is a transcript of a conversation:
        {conversation_text}

        Based on the conversation above, where the speakers and the conversation diarization in the dialogue are not very good,
        please review the conversation and try to fix the diarization by assigning the right speakers.
        Keep the same format with [Speaker X]: but make sure the speaker assignments are correct.
        """
        logging.info(f"Sending to Gemini for diarization correction (approx lines: {len(conversation_text.splitlines())})...")
        response = model.generate_content(prompt)
        corrected_text = response.text.strip()
        logging.info(f"Received correction from Gemini. Length: {len(corrected_text)}")
        return corrected_text
    except Exception as e:
        logging.error(f"Error in diarization correction: {e}", exc_info=True)
        return None

# --- Diarization Correction Thread Function (Unchanged, but added logging) ---
# Note: processing_queue argument is unused here based on original code structure.
def diarization_correction_thread(unused_processing_queue): # Explicitly mark as unused if not needed
    """Thread that periodically corrects diarization using Gemini."""
    global segment_count # Uses global segment_count modified by process_transcripts

    # Initialize the diarization model once per thread instance for efficiency
    diarization_model_instance = None
    try:
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION) # Init once
        diarization_model_instance = GenerativeModel(DIARIZATION_LLM_MODEL_NAME)
        logging.info(f"Vertex AI model '{DIARIZATION_LLM_MODEL_NAME}' loaded for diarization correction thread.")
    except Exception as e:
        logging.error(f"Could not load Vertex AI model '{DIARIZATION_LLM_MODEL_NAME}' for diarization thread: {e}", exc_info=True)
        # If the model fails to load, this thread won't be able to perform corrections.

    while True:
        time.sleep(2) # Check periodically, adjust as needed. Original was 1s.

        if not diarization_model_instance:
            logging.warning("Diarization correction model not loaded, skipping correction cycle.")
            continue # Wait and try to re-initialize or simply skip if fatal

        perform_correction = False
        current_conversation_snapshot = "" # To hold the snapshot

        with conversation_lock:
            # Check if enough new segments have been processed
            if segment_count >= SEGMENTS_BEFORE_CORRECTION and conversation_history:
                # Take a snapshot of the current conversation history for correction
                current_conversation_snapshot = "\n".join(conversation_history)
                logging.info(f"Triggering diarization correction. Segments processed since last: {segment_count}. Total history length approx: {len(current_conversation_snapshot)} chars.")
                segment_count = 0  # Reset counter immediately after deciding to correct
                perform_correction = True

        if perform_correction and current_conversation_snapshot:
            # Call LLM with the model instance (avoids re-init of vertexai and model loading)
            prompt = f"""
            The diarization of Google speech to text does not work very well. Here is a transcript of a conversation:
            {current_conversation_snapshot}

            Based on the conversation above, where the speakers and the conversation diarization in the dialogue are not very good,
            please review the conversation and try to fix the diarization by assigning the right speakers.
            Keep the same format with [Speaker X]: but make sure the speaker assignments are correct. Example: [Speaker 1]: Hello.
            """
            try:
                logging.info(f"Sending to Gemini for diarization correction (snapshot lines: {len(current_conversation_snapshot.splitlines())})...")
                gemini_response = diarization_model_instance.generate_content(prompt)
                corrected_version = gemini_response.text.strip()
                logging.info(f"Received correction from Gemini. Corrected length: {len(corrected_version)}")

                if corrected_version:
                    with diarization_correction_lock:
                        diarization_correction_history.append({
                            'original_snapshot_length': len(current_conversation_snapshot), # Storing length for reference
                            'corrected': corrected_version,
                            'timestamp': time.time()
                        })
                        # Keep history from getting too large if needed
                        # max_history_items = 10
                        # if len(diarization_correction_history) > max_history_items:
                        # diarization_correction_history = diarization_correction_history[-max_history_items:]
                        logging.info(f"Stored corrected version. Total corrections in history: {len(diarization_correction_history)}")
                        logging.debug(f"Last corrected version: \n{corrected_version[:500]}...") # Log a snippet

                    # If using SocketIO, this is where you'd emit the 'diarization_correction' event
                    # Example:
                    # if socketio_instance:
                    #     socketio_instance.emit('diarization_correction', {'corrected': corrected_version})

            except Exception as e:
                logging.error(f"Error during diarization correction LLM call: {e}", exc_info=True)
        # No else needed, if not performing_correction, just loops and sleeps.


# --- Callback for Diarization Correction (Unchanged, assumed for SocketIO context) ---
def handle_diarization_correction(data, socketio_instance):
    """Handles the corrected diarization data and emits to the client."""
    try:
        logging.info(f"Emitting diarization_correction event via SocketIO.")
        # socketio_instance.emit('diarization_correction', {
        #     'original': data.get('original'), # Or snapshot reference
        #     'corrected': data['corrected']
        # })
    except Exception as e:
        logging.error(f"Error handling diarization correction callback for emit: {e}", exc_info=True)

# --- Get Patient Info (Unchanged, added logging) ---
def get_patient_info(patient_id):
    """Get patient information from the database."""
    try:
        conn = sqlite3.connect('patients.db') # Ensure this path is correct
        cursor = conn.cursor()
        cursor.execute('''
            SELECT first_name, last_name, date_of_birth, weight, allergies, smokes, medications, last_visit_reason, last_visit_date
            FROM patients
            WHERE id = ?
        ''', (patient_id,))
        patient = cursor.fetchone()
        conn.close()
        
        if patient:
            age = None
            if patient[2]:
                try:
                    birthdate = datetime.strptime(patient[2], '%Y-%m-%d')
                    today = datetime.now()
                    age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
                except ValueError:
                    logging.warning(f"Could not parse date_of_birth: {patient[2]}")
            
            return {
                'first_name': patient[0],
                'last_name': patient[1],
                'date_of_birth': patient[2],
                'age': age,
                'weight': patient[3],
                'allergies': patient[4],
                'smokes': "Yes" if patient[5] == 1 else "No" if patient[5] == 0 else None,
                'medications': patient[6],
                'last_visit_reason': patient[7],
                'last_visit_date': patient[8]
            }
        return None
    except Exception as e:
        logging.error(f"Error getting patient info: {e}", exc_info=True)
        return None

# --- Main Execution Block (Adjusted for new logging) ---
if __name__ == "__main__":
    logging.info("Starting audio processing application (Standalone Mode)...")
    # Note: SocketIO emissions will not work directly in this mode
    # unless a dummy or actual SocketIO server is integrated here.

    # Queues are already global
    # processing_queue = queue.Queue() # This is already global

    # Start Threads
    # Pass the global processing_queue to the threads
    transcription_thread = threading.Thread(
        target=run_transcription, args=(processing_queue,), daemon=True, name="TranscriptionThread"
    )
    processing_thread = threading.Thread(
        target=process_transcripts,
        args=(processing_queue, current_patient_id),  # Pass both queue and patient_id
        daemon=True
    )
    # The diarization_correction_thread's processing_queue argument was unused, so passing it is optional
    # or it can be removed from its definition if truly not needed by its logic.
    # For now, keeping it as `unused_processing_queue` in its definition.
    diarization_thread = threading.Thread(
        target=diarization_correction_thread, args=(processing_queue,), daemon=True, name="DiarizationCorrectionThread"
    )

    transcription_thread.start()
    processing_thread.start()
    diarization_thread.start()

    stream = None
    try:
        # Check available devices if needed: print(sd.query_devices())
        # You might want to specify a device index: device=sd.default.device[0] or specific device index
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE, # This is frames per callback
            channels=1,
            dtype='int16',
            callback=audio_callback
        )
        stream.start()
        logging.info(f"Recording started from device: {stream.device} with samplerate: {stream.samplerate} Hz. Press Ctrl+C to stop.")

        while transcription_thread.is_alive() or processing_thread.is_alive() or diarization_thread.is_alive():
            time.sleep(0.5) # Main thread sleep, threads are active

    except KeyboardInterrupt:
        logging.info("\nCtrl+C received. Initiating shutdown...")
        # Signal audio generator to stop, which will end the transcription stream
        audio_buffer.put(None)
        # The transcription thread's finally block will put None into processing_queue
        # to signal the processing_thread. If process_transcripts could be blocked on
        # something else, an additional signal might be needed.

    except Exception as e:
        logging.critical(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
        # Ensure stop signals are sent even on unexpected error
        if audio_buffer.empty() or not (audio_buffer.queue and audio_buffer.queue[-1] is None): # Avoid multiple Nones if already sent
             audio_buffer.put(None)
        # The transcription_thread's finally clause should handle its queue,
        # but if it died before that, processing_queue might need a None.
        # However, this is tricky as the processing_queue is managed by transcription_thread's lifecycle.

    finally:
        if stream and stream.active:
             logging.info("Stopping audio stream...")
             stream.stop()
             stream.close()
             logging.info("Audio stream stopped and closed.")

        logging.info("Waiting for threads to finish...")
        if transcription_thread.is_alive():
            transcription_thread.join(timeout=10) # Increased timeout
            if transcription_thread.is_alive():
                logging.warning("Transcription thread did not finish in time.")
        if processing_thread.is_alive():
            # Ensure processing_queue has a None if transcription thread died unexpectedly before sending it
            # This is a safeguard; ideally, run_transcription's finally block handles it.
            # if not transcription_thread.is_alive() and (processing_queue.empty() or processing_queue.queue[-1] is not None):
            #    logging.info("Transcription thread seems dead, ensuring processing_queue gets a None.")
            #    processing_queue.put(None) # This might cause issues if transcription thread *also* puts None.
                                         # Better to rely on the transcription_thread's finally.
            processing_thread.join(timeout=10) # Increased timeout
            if processing_thread.is_alive():
                logging.warning("Processing thread did not finish in time.")

        # Diarization thread is a daemon, it will exit when non-daemon threads exit.
        # However, joining it explicitly is cleaner if it has cleanup.
        if diarization_thread.is_alive():
            # Diarization thread might be sleeping or in a long LLM call.
            # It doesn't have an explicit stop signal other than program termination.
            # For graceful shutdown, you might add a global `shutdown_event` threading.Event()
            # that it can check in its loop.
            logging.info("Diarization thread is daemon; will exit with app. No explicit join or it might block shutdown.")
            # diarization_thread.join(timeout=5) # Optionally join

        logging.info("Application finished.")

        