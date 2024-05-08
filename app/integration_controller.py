import asyncio

from .chatbot.chatbot_service import ChatbotService
from .stt.stt_service import STTService
from .tts.tts_service import TextToSpeechService


class IntegrationController:
    def __init__(
        self,
        stt_model_path,
        chatbot_model_path,
        tts_model_path,
        tts_vocoder_path,
    ):
        self.stt_service = STTService(stt_model_path)
        self.chatbot_service = ChatbotService(chatbot_model_path)
        self.tts_service = TextToSpeechService(
            tts_model_path, tts_vocoder_path
        )

    async def handle_interaction(self, audio_input):
        # Convert speech to text
        text_input = self.stt_service.transcribe_audio(audio_input)

        # Generate a response using the chatbot
        chat_response = self.chatbot_service.generate_response(text_input)

        # Convert text response to speech
        async for audio_output in self.tts_service.synthesize_stream(
            chat_response
        ):
            yield audio_output


# Example usage
async def main():
    controller = IntegrationController(
        "path_to_stt_model",
        "path_to_chatbot_model",
        "path_to_tts_model",
        "path_to_vocoder",
    )
    async for output in controller.handle_interaction("path_to_audio_input"):
        print(output)


# Running the async main function to test integration
if __name__ == "__main__":
    asyncio.run(main())
