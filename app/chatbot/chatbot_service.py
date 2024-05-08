import logging

from TTS.utils.io import load_config
from TTS.utils.synthesizers import Synthesizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TextToSpeechService:
    def __init__(self, config_path, model_path, vocoder_path):
        self.config = load_config(config_path)
        try:
            self.synthesizer = Synthesizer(
                tts_checkpoint=model_path,
                tts_config_path=config_path,
                vocoder_checkpoint=vocoder_path,
                vocoder_config=self.config,
                use_cuda=True,
            )
            logger.info("Text-to-Speech model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            raise RuntimeError(f"Text-to-Speech model loading failed: {e}")

    def synthesize_text(self, text):
        try:
            audio, _, _ = self.synthesizer.tts(text)
            logger.info("Audio synthesis successful.")
            return audio
        except Exception as e:
            logger.error(f"Error during text synthesis: {e}")
            return None


# Example usage
if __name__ == "__main__":
    tts_service = TextToSpeechService(
        config_path="config.json",
        model_path="tts_model.pth.tar",
        vocoder_path="vocoder_model.pth.tar",
    )
    audio_output = tts_service.synthesize_text(
        "Hello, how can I assist you today?"
    )
    # Handling of audio output to be added based on system requirements
