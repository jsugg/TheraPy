# app/web/run.py
import asyncio

from typing import AsyncGenerator, Any, Coroutine
from flask import Flask, Blueprint, render_template
from flask_socketio import SocketIO, emit

from app.integration_controller import IntegrationController
from app.services.chatbot import ChatbotService
from app.services.stt import SpeechToTextService
from app.services.tts import TextToSpeechService
from app.web.flask_app import create_app


# Flask app initialization
main: Blueprint = Blueprint(
    name="main", import_name=__name__, template_folder="templates"
)
app: Flask = create_app()
socketio: SocketIO = SocketIO(app=app, cors_allowed_origins="*")

# Initialize ML services
stt_service = SpeechToTextService(model_path="openai/whisper-medium")
chatbot_service = ChatbotService(model_name="microsoft/Phi-3-mini-4k-instruct")
tts_service = TextToSpeechService(
    config_path="models/tts/config.json", model_path="models/tts/"
)

# Initialize integration controller
controller = IntegrationController(
    stt_service=stt_service,
    chatbot_service=chatbot_service,
    tts_service=tts_service,
)


# Define routes
@main.route(rule="/")
def index() -> str:
    return render_template(template_name_or_list="index.html")


# Define socket events
@socketio.on(message="stream_audio")
def handle_audio_stream(data) -> None:
    audio_stream = data["audio"]

    async def process_audio() -> None:
        async for audio_chunk in controller.handle_interaction(audio_data={"audio": audio_stream}):
            emit("audio_response", {"audio": audio_chunk})

    asyncio.create_task(process_audio())


# Register routes and socket events
app.register_blueprint(blueprint=main, url_prefix="/")
socketio.init_app(app=app)

# Run the Flask app
if __name__ == "__main__":
    socketio.run(app=app, debug=True, use_reloader=False, log_output=True)
