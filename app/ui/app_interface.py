# app/ui/app_interface.py

import logging
from typing import Any, AsyncGenerator, Callable, Dict, Optional
import streamlit as st
from streamlit_webrtc import (
    WebRtcStreamerContext,
    webrtc_streamer,
    WebRtcMode,
    RTCConfiguration,
    AudioProcessorBase,
)
import asyncio
import av
from app.integration_controller import IntegrationController
from app.stt.speech_to_text_service import SpeechToTextService
from app.chatbot.chatbot_service import ChatbotService
from app.tts.text_to_speech_service import TextToSpeechService
from app.types import SingletonMeta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(name=__name__)


class AudioProcessorSingletonMeta(SingletonMeta, type(AudioProcessorBase)):
    """Combined metaclass to handle AudioProcessorBase and SingletonMeta
    metaclass conflicts."""


class AudioProcessor(
    AudioProcessorBase, metaclass=AudioProcessorSingletonMeta
):
    def __init__(self, controller: IntegrationController) -> None:
        super().__init__()
        self.controller: IntegrationController = controller
        self.loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop=self.loop)

    async def process_audio(self, audio_data):
        if audio_stream_response := (
            self.controller.handle_interaction(audio_stream=audio_data)
        ):
            return audio_stream_response

        logger.info(msg="No text detected")
        return None

    def recv(self, frame: av.AudioFrame):
        audio_data = frame.to_ndarray(format="s16le")
        return self.loop.run_until_complete(
            future=self.process_audio(audio_data=audio_data)
        )


def main() -> None:
    st.title(body="OTheraPy Voice Chatbot")
    st.header(
        body="Interactive Voice Chatbot for Adults with Asperger's Syndrome"
    )
    logger.info(msg="Starting OTheraPy Voice Chatbot application")

    # Initialize services
    stt_service = SpeechToTextService(
        # model_path="AIMH/mental-roberta-large",
        # processor_path="AIMH/mental-roberta-large"
        model_path="openai/whisper-medium"
    )
    chatbot_service = ChatbotService(
        model_name="microsoft/Phi-3-mini-4k-instruct"
    )
    tts_service = TextToSpeechService(
        config_path="models/tts/config.json",
        model_path="models/tts/",
        # "voice_conversion_models/multilingual/vctk/freevc24"
    )

    controller = IntegrationController(
        stt_service=stt_service,
        chatbot_service=chatbot_service,
        tts_service=tts_service,
    )
    logger.info(msg="Services initialized")

    if "controller" not in st.session_state:
        st.session_state["controller"] = controller
    

    def audio_processor_factory() -> Callable[[], AudioProcessor]:
        def factory() -> AudioProcessor:
            return AudioProcessor(controller=controller)
        return factory

    # Configure RTC
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_ctx: WebRtcStreamerContext[Any, AudioProcessor] = webrtc_streamer(
        key="chatbot",
        mode=WebRtcMode.SENDRECV,
        desired_playing_state=True,
        audio_processor_factory=audio_processor_factory(),
        media_stream_constraints={"video": False, "audio": True},
        async_processing=True,
    )
    logger.info(msg="WebRTC streamer initialized")


if __name__ == "__main__":
    main()
