# app.py
from flask import Flask, render_template, jsonify
import os
import threading
import queue
import sounddevice as sd
import numpy as np
import sqlite3
import random

# Import necessary functions and global variables from audio.stable.py
from audio_stable import (
    run_transcription, # Keep transcription thread function
    load_questions, # Keep load_questions for initial state route if needed
    get_current_questions, # Keep getter for initial state route
    get_transcript_segments, # Keep getter for transcript data
    get_diarization_correction_history, # Keep getter for history (won't update in this mode)
    audio_buffer, audio_callback, SAMPLE_RATE, CHUNK_SIZE, # Keep audio recording components
    process_transcripts, # Keep process_transcripts function
    diarization_correction_thread, # Add diarization correction thread
    # We will NOT be using process_transcripts, diarization_correction_thread,
    # socketio_emitter_thread, or emission_queue in this polling version.
)
import sys

app = Flask(__name__)
# Make sure SECRET_KEY is actually set in a real app, not just 'secret!'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secret_key_for_dev')

# Global variable to store current patient ID
current_patient_id = None

# Create a queue for processing
processing_queue = queue.Queue()

# --- Route for the main page (Unchanged) ---
@app.route('/')
def index():
    # We still render the same template
    return render_template('index.html')

# --- Route to get the latest transcript data ---
@app.route('/get_transcript')
def get_transcript():
    global current_patient_id
    segments = get_transcript_segments()
    questions = get_current_questions()
    corrections = get_diarization_correction_history()
    
    # Get current patient info
    patient_info = None
    if current_patient_id:
        conn = sqlite3.connect('patients.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT first_name, last_name, date_of_birth, weight, 
                   allergies, smokes, medications, last_visit_reason, last_visit_date
            FROM patients 
            WHERE id = ?
        ''', (current_patient_id,))
        patient = cursor.fetchone()
        conn.close()
        
        if patient:
            patient_info = {
                'first_name': patient[0],
                'last_name': patient[1],
                'date_of_birth': patient[2],
                'weight': patient[3],
                'allergies': patient[4],
                'smokes': patient[5],
                'medications': patient[6],
                'last_visit_reason': patient[7],
                'last_visit_date': patient[8]
            }
    
    return jsonify({
        'segments': segments,
        'questions': questions,
        'corrections': corrections,
        'patient': patient_info
    })

# --- Route to get current patient data ---
@app.route('/get_current_patient')
def get_current_patient():
    global current_patient_id
    if current_patient_id:
        conn = sqlite3.connect('patients.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT first_name, last_name, date_of_birth, weight, 
                   allergies, smokes, medications, last_visit_reason, last_visit_date
            FROM patients 
            WHERE id = ?
        ''', (current_patient_id,))
        patient = cursor.fetchone()
        conn.close()
        
        if patient:
            return jsonify({
                'first_name': patient[0],
                'last_name': patient[1],
                'date_of_birth': patient[2],
                'weight': patient[3],
                'allergies': patient[4],
                'smokes': patient[5],
                'medications': patient[6],
                'last_visit_reason': patient[7],
                'last_visit_date': patient[8]
            })
    return jsonify({'error': 'No patient selected'}), 404

# --- Route to get random patient data ---
@app.route('/get_random_patient')
def get_random_patient():
    global current_patient_id
    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    
    # Get total number of patients
    cursor.execute('SELECT COUNT(*) FROM patients')
    total_patients = cursor.fetchone()[0]
    
    # Get a random patient
    random_id = random.randint(1, total_patients)
    current_patient_id = random_id  # Update the global patient ID
    
    cursor.execute('''
        SELECT first_name, last_name, date_of_birth, weight, 
               allergies, smokes, medications, last_visit_reason, last_visit_date
        FROM patients 
        WHERE id = ?
    ''', (random_id,))
    
    patient = cursor.fetchone()
    conn.close()
    
    if patient:
        return jsonify({
            'first_name': patient[0],
            'last_name': patient[1],
            'date_of_birth': patient[2],
            'weight': patient[3],
            'allergies': patient[4],
            'smokes': patient[5],
            'medications': patient[6],
            'last_visit_reason': patient[7],
            'last_visit_date': patient[8]
        })
    else:
        return jsonify({'error': 'No patient found'}), 404

# --- Function to start audio processing threads (MODIFIED for polling) ---
def start_audio_processing():
    """Starts the transcription thread."""
    global current_patient_id
    print("Starting audio processing threads (Polling Approach)...")
    
    # Ensure we have a patient ID
    if current_patient_id is None:
        # Get initial random patient ID if not set
        conn = sqlite3.connect('patients.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM patients')
        total_patients = cursor.fetchone()[0]
        current_patient_id = random.randint(1, total_patients)
        conn.close()
        print(f"Selected initial patient ID: {current_patient_id}")

    # Create a queue for communication between transcription and processing
    processing_queue = queue.Queue()

    # Start the transcription thread, passing the queue.
    transcription_thread = threading.Thread(
        target=run_transcription,
        args=(processing_queue,), # Pass only the queue
        daemon=True # Thread will exit when main program exits
    )

    # Start the processing thread
    processing_thread = threading.Thread(
        target=process_transcripts,
        args=(processing_queue, current_patient_id), # Pass the queue and current patient ID
        daemon=True
    )

    # Start the diarization correction thread
    correction_thread = threading.Thread(
        target=diarization_correction_thread,
        args=(processing_queue,), # Pass only the queue
        daemon=True
    )

    transcription_thread.start()
    processing_thread.start()
    correction_thread.start()

    print("All audio processing threads started.")

# --- Main execution block (Modified for Polling Approach) ---
if __name__ == '__main__':
    print("Starting Flask app with Audio Processing (Polling Approach)...")

    # Load questions at startup in the main process
    load_questions()

    # Get initial random patient ID
    conn = sqlite3.connect('patients.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM patients')
    total_patients = cursor.fetchone()[0]
    current_patient_id = random.randint(1, total_patients)
    conn.close()
    print(f"Selected initial patient ID: {current_patient_id}")

    # Start only the transcription thread
    start_audio_processing()

    # Start the audio stream recording
    stream = None
    try:
        # sd.InputStream callback runs in a separate thread managed by sounddevice
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            channels=1,
            dtype='int16',
            callback=audio_callback # This callback just puts audio into audio_buffer
        )
        print("Starting audio recording stream...")
        stream.start()
        print(f"Recording started with sample rate {SAMPLE_RATE}, chunk size {CHUNK_SIZE}. Press Ctrl+C to stop.")

        # Run the Flask app
        # We are NOT using socketio.run()
        app.run(debug=True, use_reloader=False) # use_reloader=False needed for threading
        print("Flask server stopped.") # This line will be reached when the server stops

    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping recording and processing...")
        # Signal threads to stop gracefully
        audio_buffer.put(None) # Stops audio_generator -> stops run_transcription
        # No need to signal other queues/threads as they are not running

    except Exception as e:
        print(f"\nAn unexpected error occurred during startup or main loop: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Ensure threads are signaled to stop even on other errors
        if audio_buffer.empty(): audio_buffer.put(None)


    finally:
        # Cleanup
        if stream and stream.active:
             print("Stopping audio stream...")
             stream.stop()
             stream.close()

        print("Waiting for threads to finish...")
        # Add joins here if necessary for cleanup
        # transcription_thread.join(timeout=5)


        print("Application finishing.")
