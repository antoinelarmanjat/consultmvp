# audio.py (with Diarization and Vertex AI LLM Question Monitoring)
import time
import queue
import sys
import threading
import os
from threading import Lock
import sqlite3
from datetime import datetime

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
CHUNK_SIZE = int(SAMPLE_RATE / 10)
LANGUAGE_CODE = "en-US"
EXPECTED_SPEAKERS = 2

# --- Vertex AI Configuration ---
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
processing_queue = queue.Queue()
questions_list = []  # Global questions list
questions_lock = threading.Lock()  # Lock for thread-safe access to questions_list
questions_loaded = False  # Flag to track if questions have been loaded
conversation_history = [] # Used by LLM processing
conversation_lock = Lock() # New: Also store segments for initial client load, updated by process_transcripts
transcript_segments = []
transcript_lock = Lock()

diarization_correction_history = [] # Store corrected versions
diarization_correction_lock = Lock()
segment_count = 0 # Count segments processed by process_transcripts
SEGMENTS_BEFORE_CORRECTION = 7

# Pass the socketio instance globally or pass to functions that need it
# Passing to functions is cleaner. Let's pass it to threads.

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
                                'general': len(questions_list) < 5  # First 5 questions are general
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
        a=1
    audio_buffer.put(bytes(indata))

# --- Generator for Audio Stream to API (Unchanged) ---
def audio_generator():
    while True:
        chunk = audio_buffer.get()
        if chunk is None:
            break
        yield speech.StreamingRecognizeRequest(audio_content=chunk)

# --- Speech-to-Text Thread Function (MODIFIED to emit new_transcript) ---
# Now accepts socketio instance to emit directly
# --- Speech-to-Text Thread Function (MODIFIED for extensive logging) ---
# Now accepts socketio instance to emit directly
def run_transcription(processing_queue):
    client = speech.SpeechClient()

    # --- Configuration ---
    # Keep your original config for now, but be ready to simplify if needed
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
    stream_ended_normally = False

    try:
        responses = client.streaming_recognize(
            config=streaming_config,
            requests=requests,
        )
        #print("[Transcription Thread] Receiving responses from Google API...")

        # --- Log each response received ---
        # Iterate through responses. This loop blocks until a response is available.
        for i, response in enumerate(responses):
            # Use sys.stdout.flush() to ensure prints appear immediately,
            # especially with the '\r' from the interim print.
            sys.stdout.flush()

            if not response.results:
                #print("-> No results in response.")
                continue

            result = response.results[0]
            #print(f"-> is_final: {result.is_final}.", end=' ')
            sys.stdout.flush()

            if not result.alternatives:
                print("-> No alternatives in result.")
                continue

            # Always print interim results for user feedback
            if not result.is_final:
                interim_transcript = result.alternatives[0].transcript
                #print(f"Interim: {interim_transcript}", end='\r') # Use \r to overwrite line
                sys.stdout.flush()
                continue # Go to the next response

            # --- This block is ONLY reached if result.is_final is True ---
            # Overwrite the interim line with the final segment info
            #print(f"\n[Transcription Thread] Final Segment {i+1} received.")
            sys.stdout.flush()

            final_alternative = result.alternatives[0]
            if not final_alternative.words:
                print("[Transcription Thread] Final result has no words.")
                continue

            # --- Diarization/Formatting (Only if diarization is enabled) ---
            # Check if diarization was enabled in the config sent to Google
            if hasattr(recognition_config, 'diarization_config') and recognition_config.diarization_config.enable_speaker_diarization:
                 diarized_segment = ""
                 current_speaker_tag = None
                 #print("[Transcription Thread] Processing words with diarization tags:")
                 for word_info in final_alternative.words:
                     tag = word_info.speaker_tag if hasattr(word_info, 'speaker_tag') else 0 # Default to 0 if tag is missing for some reason
                     word_text = word_info.word

                     if current_speaker_tag is None or tag != current_speaker_tag:
                         if current_speaker_tag is not None:
                            diarized_segment += "\n" # Add newline between speaker changes
                         current_speaker_tag = tag
                         diarized_segment += f"[Speaker {current_speaker_tag}]: {word_text}"
                     else:
                         diarized_segment += f" {word_text}"
                 #print("[Transcription Thread] Finished processing words.")

            else:
                 # If diarization is disabled, just get the raw transcript
                 diarized_segment = final_alternative.transcript
                 #print("[Transcription Thread] Diarization disabled, using raw transcript.")

            # Print only the diarized final result
            print("\n[Transcription Thread] --- FINAL DIARIZED SEGMENT ---")
            print(diarized_segment)
            print("------------------------------------")

            # --- Process and Emit the Final Segment ---
            print(f"[DEBUG-EMIT] Attempting to put segment in processing queue...")
            processing_queue.put(diarized_segment)
            print("[DEBUG-EMIT] Segment put in queue.")

            #print("[Transcription Thread] Finished processing final segment.")


        # --- End of the 'for response in responses:' loop ---
        # This line is only reached if the 'responses' iterator finishes without error (e.g., audio_generator yielded None)
        stream_ended_normally = True
        #print("\n[Transcription Thread] 'responses' iterator finished.")

    except Exception as e:
        # Catch any exception that occurs during the streaming or processing loop
        #print(f"\n[Transcription Thread] !!! FATAL ERROR during transcription stream: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # Print the full traceback for debugging

    finally:
        # This block always runs when the try or except block finishes
        #print(f"\n[Transcription Thread] Transcription stream thread stopping. Stream ended normally: {stream_ended_normally}.")
        # Signal stop to processing threads. Putting None multiple times is safe.
        #rint("[Transcription Thread] Signaling processing queue to stop.")
        processing_queue.put(None) # Stop the processing thread


# --- Transcript Processing Thread Function (MODIFIED to remove callback and emit question_update) ---
# Now accepts socketio instance to emit question_update directly
def process_transcripts(processing_queue, patient_id):
    """Runs in a separate thread, processing segments, managing questions, and updating history."""
    global segment_count, conversation_history, transcript_segments

    print("Processing thread started, loading questions...")
    load_questions() # Load questions at the start
    print("QUESTIONS LIST",questions_list)

    # Initialize Vertex AI
    try:
        print(f"Initializing Vertex AI for Project '{GCP_PROJECT_ID}' in Location '{GCP_LOCATION}'...")
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    except Exception as e:
        print(f"ERROR: Failed to initialize Vertex AI: {e}")
        return

    # Load Vertex AI Model for questions
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    try:
        model = GenerativeModel(LLM_MODEL_NAME)
        print(f"Vertex AI model '{LLM_MODEL_NAME}' loaded for questions.")
    except Exception as e:
        print(f"ERROR: Could not load Vertex AI model '{LLM_MODEL_NAME}' for questions: {e}")
        return

    # --- Initial Question Analysis (Moved from while loop) ---
    print("\n" + "="*15 + " Initial General Questions " + "="*15)
    
    # Get patient info from database using the provided patient_id
    patient_info = get_patient_info(patient_id)
    
    if patient_info:
        print(f"Found patient info: {patient_info}")
        print(f"FOUND QUESTIONS LIST: {questions_list}")
        
        # Create a prompt for the LLM to analyze which questions are already answered
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
            print("ANALYSIS", analysis)
            
            # Clean up the response by removing markdown code block markers
            if analysis.startswith('```json'):
                analysis = analysis[7:]  # Remove ```json
            if analysis.endswith('```'):
                analysis = analysis[:-3]  # Remove ```
            analysis = analysis.strip()  # Remove any extra whitespace
            
            # Parse the JSON response
            import json
            try:
                answers = json.loads(analysis)
                
                # Update questions based on the answers
                for answer in answers:
                    question_text = answer['question']
                    answer_text = answer['answer']
                    
                    # Find and update the question in our list
                    for question in questions_list:
                        if question['text'] == question_text and question['status'] == 'pending':
                            if answer_text != "needs to be asked":
                                question['status'] = 'answered'
                                question['answer'] = answer_text
                                print(f"Question already answered from database: {question_text}")
                                print(f"Answer: {answer_text}")
                            else:
                                question['status'] = 'suggested'
                                print(f"Question needs to be asked: {question_text}")
                            break
            
            except json.JSONDecodeError as e:
                print(f"Error parsing LLM response as JSON: {e}")
                print(f"Raw response after cleanup: {analysis}")
                # Fallback to suggesting all questions if JSON parsing fails
                for question in questions_list:
                    if question['general'] and question['status'] == 'pending':
                        question['status'] = 'suggested'
        
        except Exception as e:
            print(f"Error analyzing database info with LLM: {e}")
            # Fallback to suggesting all questions if LLM analysis fails
            for question in questions_list:
                if question['general'] and question['status'] == 'pending':
                    question['status'] = 'suggested'
    else:
        print("No patient info found in database, suggesting all general questions")
        # If no patient info, suggest all general questions
        for question in questions_list:
            if question['general'] and question['status'] == 'pending':
                question['status'] = 'suggested'
    
    print("="*51 + "\n")
    initial_suggestions_done = True

    # --- Main Processing Loop ---
    while True:
        segment = processing_queue.get()

        if segment is None:
            print("Processing thread received stop signal.")
            break

        print(f"\n[Processor] Processing Segment:\n{segment}\n-----------------------------")

        # Update conversation history
        with conversation_lock:
            conversation_history.append(segment)
            segment_count += 1

        # Update transcript segments
        with transcript_lock:
            transcript_segments.append(segment)

        # --- 2. Check for Answers (Unchanged logic, emit question_update after) ---
        # Process this segment to see if it answers any *currently suggested* questions
        updated_questions_during_segment_processing = False
        print("QUESTIONS TO CHECK",questions_list)
        with questions_lock:
            # Check questions that are 'suggested'
            questions_to_check = [q for q in questions_list if q['status'] == 'suggested']
            for question in questions_to_check:
                # Use the full context for answer checking (can optimize later if needed)
                full_context = "\n".join(conversation_history)
                prompt = f"""
                This is a conversation transcript:
                ---
                {full_context}
                ---
                Has the question "{question['text']}" been answered or addressed in the MOST RECENT part of the conversation, or earlier?
                Dont pay any attention to the diarization because it does not work, try to figure out the answers without
                taking care of the Speaker indications.
                Respond in one of two ways EXACTLY:
                1. If YES (or likely), respond with: YES - [The specific answer extracted from the conversation]
                2. If NO, respond with: NO
                """
                try:
                    response = model.generate_content(prompt, safety_settings=safety_settings)
                    llm_answer = response.text.strip()
                    # print(f"Answer check for question '{question['text']}': {llm_answer}") # Optional debug print

                    if llm_answer.startswith("YES -"):
                        extracted_answer = llm_answer[len("YES -"):].strip()
                        # Only update if status changes or answer is new/different
                        if question['status'] != 'answered' or question['answer'] != extracted_answer:
                            question['status'] = 'answered'
                            question['answer'] = extracted_answer
                            updated_questions_during_segment_processing = True
                            print("="*40)
                            print(f"[System Found Answer]")
                            print(f"  Q: {question['text']}")
                            print(f"  A: {extracted_answer}")
                            print("="*40 + "\n")
                            # print(f"Updated question: {question['text']} (status: {question['status']}, answer: {question['answer']})") # Optional debug print

                except Exception as e:
                    print(f"[Processor] ERROR during Vertex AI API call for answer check '{question['text']}': {e}")

        # --- 3. Ongoing Contextual Question Suggestions (Unchanged logic, emit question_update after) ---
        # Suggest new questions based on the full context
        updated_suggestions_during_segment_processing = False
        if initial_suggestions_done: # Ensure initial ones are handled first
            with questions_lock:
                # Only suggest from 'pending' contextual questions
                pending_contextual_questions = [
                    q for q in questions_list if not q['general'] and q['status'] == 'pending'
                ]
                if pending_contextual_questions:
                    full_context = "\n".join(conversation_history)
                    candidate_question_texts = [q['text'] for q in pending_contextual_questions]
                    # Using the same prompt, potentially optimize context length later
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
                        # print(f"Suggested questions from Gemini: {suggested_lines}") # Optional debug print

                        if suggested_lines and suggested_lines[0].strip().lower() != "none":
                            print("\n[System Suggests Asking (Contextual)]:")
                            for suggested_text in suggested_lines:
                                suggested_text = suggested_text.strip()
                                if not suggested_text: continue

                                # Find the question in our list and update its status
                                for question in pending_contextual_questions:
                                    # Use a robust check (lower case, possibly fuzzy match if needed)
                                    if question['text'].strip().lower() == suggested_text.lower() or suggested_text.lower() in question['text'].strip().lower():
                                        if question['status'] == 'pending':
                                            print(f"  - {question['text']}")
                                            question['status'] = 'suggested'
                                            updated_suggestions_during_segment_processing = True
                                            # print(f"Updated question: {question['text']} (status: {question['status']})") # Optional debug print
                                        break # Found the match, move to next suggested_text

                    except Exception as e:
                        print(f"[Processor] ERROR during Vertex AI API call for question suggestion: {e}")
                # else:
                #      print("[Processor]: No pending contextual questions to suggest.") # Optional debug print

    # Final status print (unchanged)
    print("\n--- Final Question Status ---")
    with questions_lock:
        for q in questions_list:
            print(f"Q: {q['text']}")
            print(f"  Status: {q['status']}")
            if q['answer']:
                print(f"  Answer: {q['answer']}")
    print("---------------------------\n")
    print("Processing thread finished.")


# --- Diarization Correction Helper (MODIFIED to use passed socketio_instance) ---
def send_to_gemini_for_diarization(conversation_text):
    """Send conversation to Gemini for diarization correction."""
    try:
        # Load Vertex AI Model for correction if not already loaded or use a different one
        # Note: Re-initializing/loading model per call is inefficient, but kept for simplicity
        # If performance is critical, load this model once in the thread setup.
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        model = GenerativeModel(DIARIZATION_LLM_MODEL_NAME) # Use the specific diarization model name
        prompt = f"""
        The diarization of Google speech to text does not work very well. Here is a transcript of a conversation:
        {conversation_text}

        Based on the conversation above, where the speakers and the conversation diarization in the dialogue are not very good,
        please review the conversation and try to fix the diarization by assigning the right speakers.
        Keep the same format with [Speaker X]: but make sure the speaker assignments are correct.
        """

        print(f"Sending to Gemini for diarization correction (segments: {len(conversation_text.splitlines())})...")
        response = model.generate_content(prompt)
        print(f"Received correction from Gemini.") # Avoid printing full correction here, can be long
        return response.text.strip()
    except Exception as e:
        print(f"Error in diarization correction: {e}")
        return None

# --- Diarization Correction Thread Function (MODIFIED to pass socketio_instance to callback) ---
# Accepts socketio instance
def diarization_correction_thread(processing_queue):
    """Thread that periodically corrects diarization using Gemini."""
    global segment_count

    # Optional: Initialize the diarization model here if not doing it per call
    # try:
    #     vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
    #     diarization_model = GenerativeModel(DIARIZATION_LLM_MODEL_NAME)
    #     print(f"Vertex AI model '{DIARIZATION_LLM_MODEL_NAME}' loaded for diarization.")
    # except Exception as e:
    #     print(f"ERROR: Could not load Vertex AI model '{DIARIZATION_LLM_MODEL_NAME}' for diarization: {e}")
    #     diarization_model = None # Handle case where model load fails

    while True:
        time.sleep(1)  # Check every second

        with conversation_lock:
            # Only trigger if we have processed enough new segments AND there's history
            if segment_count >= SEGMENTS_BEFORE_CORRECTION and conversation_history:
                # Get current conversation
                current_conversation = "\n".join(conversation_history)
                print(f"Triggering diarization correction after {segment_count} new segments.")

                # Reset counter BEFORE the potentially long LLM call
                segment_count = 0

                # Call the LLM to get corrected version
                # Pass only the conversation text here, LLM logic is inside the helper
                corrected_version = send_to_gemini_for_diarization(current_conversation)

                if corrected_version:
                    with diarization_correction_lock:
                        # Store the corrected version along with the original it was based on
                        diarization_correction_history.append({
                            'original': current_conversation,
                            'corrected': corrected_version
                        })
                        print(f"Stored corrected version {diarization_correction_history}.")

                    # Call the callback to update the web interface, passing socketio_instance
                    """
                    callback({
                        'original': current_conversation,
                        'corrected': corrected_version
                    }, socketio_instance) # Pass socketio instance to the callback
                    """
# --- Callback for Diarization Correction (MODIFIED to accept socketio) ---
# Moved this handler function here as it's tightly coupled with the correction thread logic.
# It needs the socketio instance to emit.
def handle_diarization_correction(data, socketio_instance):
    """Handles the corrected diarization data and emits to the client."""
    try:
        # We already stored the history in the correction thread, just emit the latest
        print(f"Emitting diarization_correction event.")
        """
        try:
             socketio_instance.emit('diarization_correction', {
                 'original': data['original'],
                 'corrected': data['corrected']
             })
        except Exception as e:
             print(f"Error emitting diarization_correction: {e}", file=sys.stderr)
        """
    except Exception as e:
        print(f"Error handling diarization correction callback: {e}", file=sys.stderr)

def get_patient_info(patient_id):
    """Get patient information from the database."""
    try:
        conn = sqlite3.connect('patients.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT first_name, last_name, date_of_birth, weight, allergies, smokes, medications, last_visit_reason, last_visit_date
            FROM patients
            WHERE id = ?
        ''', (patient_id,))
        patient = cursor.fetchone()
        conn.close()
        
        if patient:
            # Calculate age if birthdate is present
            age = None
            if patient[2]:  # date_of_birth
                birthdate = datetime.strptime(patient[2], '%Y-%m-%d')
                today = datetime.now()
                age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
            
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
        print(f"Error getting patient info: {e}")
        return None

# --- Main Execution Block (Adjusted for passing socketio if run standalone) ---
# Note: This __main__ block is typically not used when run via app.py,
# but kept for completeness if you wanted to test audio.py directly (without Flask/SocketIO)
if __name__ == "__main__":
    print("Starting audio processing application with Diarization and Vertex AI LLM Monitoring (Standalone Mode)...")
    print("Note: SocketIO emissions will not work in this mode.")

    """# Create dummy socketio instance if running standalone, to avoid errors
    class DummySocketIO:
        def emit(self, event, data):
            print(f"[DummySocketIO] Emitted event '{event}' with data (truncated): {str(data)[:100]}...")
        def start_background_task(self, target, *args, **kwargs):
             # In standalone, just run the task directly or in a simple thread
             print(f"[DummySocketIO] Starting background task: {target.__name__}")
             thread = threading.Thread(target=target, args=args, kwargs=kwargs)
             thread.daemon = True
             thread.start()
             return thread

    dummy_socketio = DummySocketIO()
"""
    # --- Start Threads ---
    processing_queue = queue.Queue()
    # Pass dummy_socketio to threads
    transcription_thread = threading.Thread(target=run_transcription, args=(processing_queue,), daemon=True)
    processing_thread = threading.Thread(target=process_transcripts, args=(processing_queue, 1), daemon=True) # Pass socketio here
    diarization_thread = threading.Thread(target=diarization_correction_thread, args=(processing_queue,), daemon=True) # Pass socketio here

    transcription_thread.start()
    processing_thread.start()
    diarization_thread.start()

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

        # Keep main thread alive
        while transcription_thread.is_alive() or processing_thread.is_alive() or diarization_thread.is_alive():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping recording and processing...")
        audio_buffer.put(None) # Signal audio generator to stop
        processing_queue.put(None) # Signal processing thread to stop

    except Exception as e:
        print(f"\nAn unexpected error occurred in the main loop: {e}")
        # Ensure stop signals are sent even on error
        if audio_buffer.empty(): audio_buffer.put(None)
        if processing_queue.empty(): processing_queue.put(None)

    finally:
        # Cleanup
        if stream and stream.active:
             print("Stopping audio stream...")
             stream.stop()
             stream.close()

        print("Waiting for threads to finish...")
        # Give threads a moment to shut down
        transcription_thread.join(timeout=5)
        processing_thread.join(timeout=5)
        diarization_thread.join(timeout=5)

        print("All threads finished.")
        print("Application finished.")