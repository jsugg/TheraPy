import unittest
from app.tts.text_to_speech_service import TextToSpeechService


class TestTextToSpeechService(unittest.TestCase):
    def test_synthesis(self):
        tts_service = TextToSpeechService(
            config_path="tts_config.json",
            model_path="tts_model.pth.tar",
            vocoder_path="vocoder_model.pth.tar",
        )
        result = tts_service.synthesize_text("Hello")
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
