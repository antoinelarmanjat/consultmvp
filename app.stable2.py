# app.py
from flask import Flask, render_template, jsonify # Import jsonify
from flask_socketio import SocketIO
import os
# Import necessary functions from audio.py
from audio import (
    run_transcription, process_transcripts, diarization_correction_thread,
    load_questions, get_current_questions, get_transcript_segments, # Import new getters
    get_diarization_correction_history, handle_diarization_correction # Import handle_diarization_correction
)
import threading
import queue
import sounddevice as sd
import numpy as np
# Import audio buffer and config from audio.py
from audio import audio_buffer, audio_callback, SAMPLE_RATE, CHUNK_SIZE
import sys

app = Flask(__name__)
# Make sure SECRET_KEY is actually set in a real app, not just 'secret!'
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'a_default_secret_key_for_dev')
socketio = SocketIO(app)

# Global variables for transcript storage - kept for get_transcript route and initial load
# These are now primarily updated by the process_transcripts thread
transcript_segments = [] # Managed by audio.py process_transcripts thread
# corrected_transcript = "" # No longer needed here, managed by audio.py and history list
# corrected_transcript_lock = threading.Lock() # No longer needed here

# --- Route for the main page (Unchanged) ---
@app.route('/')
def index():
    return render_template('index.html')

# --- Route to get initial state (MODIFIED to use getter functions from audio.py) ---
@app.route('/get_transcript')
def get_transcript():
    # Get data safely from audio.py's global states
    questions = get_current_questions()
    segments = get_transcript_segments() # Use the getter
    # corrected_versions = get_diarization_correction_history() # Optional: Send history on load

    # For initial load, we only send the basic transcript and questions.
    # The client JS will receive history of corrections via socket events.
    # Sending the full correction history on load might be too much data.
    # Let's just send the current segments and questions.

    # Note: This route provides the state at the time of the HTTP request.
    # Real-time updates happen via SocketIO after the connection is established.
    print("Serving initial state via /get_transcript...")
    return jsonify({ # Use jsonify for proper JSON response
        'transcript': segments, # Send all collected segments
        'questions': questions
        # Diarization correction history is not sent initially via REST, only via SocketIO events
        # 'corrected_transcript': corrected_transcript, # Removed
    })

# --- SocketIO connect handler (MODIFIED to use getter functions) ---
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send initial state after connection
    print("Sending initial state to new client via SocketIO...")

    # Send current questions state
    current_questions = get_current_questions()
    socketio.emit('question_update', {'questions': current_questions})
    print(f"Emitted initial question state ({len(current_questions)} questions).")

    # Send all existing transcript segments
    existing_segments = get_transcript_segments() # Use the getter
    if existing_segments:
        print(f"Sending {len(existing_segments)} existing transcript segments...")
        # Send them one by one or as a batch. Sending one by one simulates the real-time flow better initially.
        for segment in existing_segments:
            socketio.emit('new_transcript', {'segment': segment})
        print("Finished emitting existing transcript segments.")

    # Optional: Send existing diarization correction history on connect
    # corrected_history = get_diarization_correction_history()
    # if corrected_history:
    #      print(f"Sending {len(corrected_history)} existing correction versions...")
    #      # Send the history or just the latest correction
    #      # Sending history requires changes in client JS to handle receiving history on connect
    #      # For now, let's assume client gets them via the correction events triggered by the thread.


# --- Removed the process_transcript_segment function ---
# Its logic (appending segment and emitting new_transcript/question_update)
# has been moved directly into the audio.py threads.

# --- Diarization correction handler (MODIFIED - now receives socketio_instance) ---
# This function is called by the diarization_correction_thread in audio.py
# It now receives the socketio instance and uses it to emit.
# Moved implementation to audio.py for clarity, import and use it here.
# from audio import handle_diarization_correction # Already imported above


# --- Function to start audio processing threads (MODIFIED to pass socketio) ---
def start_audio_processing(socketio_instance):
    """Starts the transcription, processing, and diarization threads."""
    print("Starting audio processing threads...")
    # Create a queue for communication between transcription and processing
    processing_queue = queue.Queue()

    # Start the transcription thread, passing the queue and socketio instance
    transcription_thread = threading.Thread(
        target=run_transcription,
        args=(processing_queue, socketio_instance), # Pass socketio_instance
        daemon=True # Thread will exit when main program exits
    )

    # Start the processing thread, passing the queue and socketio instance
    # Note: The original 'process_transcript_segment' callback is removed.
    # question_update emission is now done directly inside process_transcripts.
    processing_thread = threading.Thread(
        target=process_transcripts,
        args=(processing_queue, socketio_instance), # Pass socketio_instance
        daemon=True
    )

    # Start the diarization correction thread, passing its callback and socketio instance
    correction_thread = threading.Thread(
        target=diarization_correction_thread,
        args=(handle_diarization_correction, socketio_instance), # Pass the callback and socketio_instance
        daemon=True
    )

    transcription_thread.start()
    processing_thread.start()
    correction_thread.start()

    print("Audio processing threads started.")

# --- Main execution block (MODIFIED to call start_audio_processing with socketio) ---
if __name__ == '__main__':
    print("Starting Flask app with SocketIO, Audio Processing, and LLM Monitoring...")

    # Load questions at startup in the main process
    load_questions()

    # Start audio processing threads, passing the socketio instance
    start_audio_processing(socketio)

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

        # Start the Flask app with SocketIO
        # socketio.run handles thread management for Flask and SocketIO
        socketio.run(app, debug=True, use_reloader=False, allow_unsafe_werkzeug=True) # use_reloader=False needed for threading

    except KeyboardInterrupt:
        print("\nCtrl+C received. Stopping recording and processing...")
        # Signal threads to stop gracefully
        audio_buffer.put(None) # Stops audio_generator -> stops run_transcription
        # processing_queue.put(None) # run_transcription's finally block already does this

    except Exception as e:
        print(f"\nAn unexpected error occurred during startup or main loop: {e}")
        import traceback
        traceback.print_exc()
        # Ensure threads are signaled to stop even on other errors
        if audio_buffer.empty(): audio_buffer.put(None)
        if processing_queue.empty(): processing_queue.put(None)


    finally:
        # Cleanup audio stream
        if stream and stream.active:
             print("Stopping audio stream...")
             stream.stop()
             stream.close()

        # Threads are daemon=True, so they should exit when the main process exits (after socketio.run returns)
        # Explicitly joining here is usually only needed if they were not daemon or if you need to ensure
        # they finish their cleanup logic before the program completely exits.
        # For this app, relying on daemon=True is common. Adding joins with timeout can be safer.
        # print("Waiting for threads to finish...")
        # # Add joins here if necessary, similar to audio.py's __main__ block

        print("Application finishing.")