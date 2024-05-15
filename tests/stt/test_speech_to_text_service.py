import unittest
from src.services.stt.speech_to_text_service import SpeechToTextService


class TestSpeechToTextService(unittest.TestCase):
    def setUp(self):
        self.service = SpeechToTextService()

    def test_transcription_quality(self):
        # Assuming a mock audio file path and expected output
        audio_path = "mock_audio.wav"
        expected_transcription = "Hello world"
        transcription = asyncio.run(
            self.service.transcribe_continuous(audio_path)
        )
        self.assertEqual(transcription, expected_transcription)


if __name__ == "__main__":
    unittest.main()
