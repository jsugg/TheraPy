# app/integration_controller.py

import asyncio
import logging
from typing import Any, AsyncGenerator, Union
from app.stt.speech_to_text_service import SpeechToTextService
from app.tts.text_to_speech_service import TextToSpeechService
from app.chatbot.chatbot_service import ChatbotService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(name=__name__)


class IntegrationController:
    def __init__(self, stt_service, chatbot_service, tts_service) -> None:
        logger.info(
            msg=(
                "Initializing IntegrationController with STT, "
                "Chatbot, and TTS services."
            )
        )
        self.stt_service: SpeechToTextService = stt_service
        self.chatbot_service: ChatbotService = chatbot_service
        self.tts_service: TextToSpeechService = tts_service
        self.loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop=self.loop)
        
        logger.info(
            msg="IntegrationController initialized."
        )

    async def handle_interaction(
        self, audio_stream
    ) -> AsyncGenerator[Any, None]:
        logger.info(msg="Starting interaction handling process.")
        try:
            transcription: Union[str, None] = (
                await self.stt_service.transcribe_until_pause(
                    audio_stream=audio_stream
                )
            )
            if transcription:
                logger.info(msg=f"Transcription: {transcription}")
                response_text: str = self.chatbot_service.generate_response(
                    transcription
                )
                logger.info(msg=f"Generated response: {response_text}")
                yield self.tts_service.synthesize_text(text=response_text)
                # async for audio_chunk in self.tts_service.synthesize_text(
                #     text=response_text
                # ):
                #     yield audio_chunk
                #     logger.info(msg="Audio chunk synthesized and yielded.")
        except Exception as e:
            logging.error(msg=f"Failed to handle interaction: {e}")
            yield None

    def cleanup(self) -> None:
        logger.info(msg="Cleaning up resources.")
        try:
            if self.loop.is_running():
                logger.info(msg="Stopping the event loop.")
                self.loop.stop()
                logger.info(msg="Event loop stopped.")
        except RuntimeError:
            logger.warning(msg="Event loop was already stopped.")
        finally:
            logger.info(msg="Cleanup completed.")
