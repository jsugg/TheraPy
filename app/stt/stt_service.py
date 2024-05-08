import asyncio
import whisper
import soundfile as sf


class SpeechToTextService:
    _instance = None  # Singleton implementation

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SpeechToTextService, cls).__new__(cls)
            device = "cuda" if whisper.cuda.is_available() else "cpu"
            cls._instance.model = whisper.load_model(
                "medium", device=device
            )  # Using 'medium' for better performance
        return cls._instance

    async def transcribe_continuous(self, audio_stream):
        # Handle continuous audio stream
        loop = asyncio.get_running_loop()
        while True:
            audio_data = await audio_stream.read()
            if not audio_data:
                break  # Stop if no more audio or silence detected
            transcription = await loop.run_in_executor(
                None, self.process_transcription, audio_data
            )
            yield transcription

    def process_transcription(self, audio_data):
        # Process audio data
        audio, rate = sf.read(audio_data)
        if rate != self.model.sample_rate:
            audio = whisper.resample(audio, rate, self.model.sample_rate)
        result = self.model.transcribe(audio)
        return result["text"]


# Example usage with continuous input (mocked for illustration)
async def main():
    stt_service = SpeechToTextService()
    async for transcription in stt_service.transcribe_continuous(
        fake_audio_stream
    ):
        print(transcription)


if __name__ == "__main__":
    asyncio.run(main())
