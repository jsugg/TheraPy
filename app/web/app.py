import io
import threading
import logging
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import numpy as np
from numpy.typing import NDArray
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from transformers.models.whisper.modeling_whisper import WhisperModel
from app.services.stt import SpeechToTextService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger: logging.Logger = logging.getLogger(name=__name__)
stt_service = SpeechToTextService(model_path="openai/whisper-medium")

app = Flask(import_name=__name__)
app.config["SECRET_KEY"] = "secret!"
socketio: SocketIO = SocketIO(app=app)

# Load Whisper model
model: WhisperModel = stt_service.model
lock: threading.Lock = threading.Lock()


def transcribe_audio(audio_chunk) -> str:
    waveform: NDArray[np.float32] = np.array(
        audio_chunk.get_array_of_samples(), dtype=np.float32
    )
    waveform /= np.iinfo(audio_chunk.sample_width * 8).max
    result = model.transcribe(waveform)
    return result["text"]


audio_buffers: dict[str, AudioSegment] = {}


@socketio.on(message="audio_message")
def handle_audio_stream(msg) -> None:
    user_id = request.sid
    logger.info(msg=f"Received audio message from {user_id}, size: {len(msg)}")
    audio_segment = None
    
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(msg), format="webm")
    except CouldntDecodeError as e:
        logger.error(msg=f"Error processing audio file: {e}")
        return

    # Ensure thread safety when accessing/modifying shared audio buffer
    with lock:
        if user_id not in audio_buffers:
            audio_buffers[user_id] = AudioSegment.empty()
        if audio_segment:
            audio_buffers[user_id] += audio_segment
            logger.info(msg=f"Buffer size now: {len(audio_buffers[user_id])}")
            process_and_transcribe_audio(user_id=user_id)


def process_and_transcribe_audio(user_id):
    with lock:  # Ensure thread safety when processing audio data
        current_buffer: AudioSegment = audio_buffers[user_id]
        if len(current_buffer) >= 5000:  # Process buffer if it's long enough
            transcript = transcribe_audio(audio_chunk=current_buffer)
            emit("transcription", transcript, to=user_id)
            logger.info(f"Transcription: {transcript}")
            audio_buffers[user_id] = (
                AudioSegment.empty()
            )  # Clear buffer after processing


@app.route("/")
def index() -> str:
    return render_template("index2.html")


if __name__ == "__main__":
    socketio.run(app=app, debug=True, use_reloader=False, log_output=True)
