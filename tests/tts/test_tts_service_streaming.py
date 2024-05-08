import pytest
import asyncio
from unittest.mock import AsyncMock
from text_to_speech_service_streaming import TextToSpeechService

@pytest.mark.asyncio
async def test_streaming_synthesize():
    # Create an instance of the TextToSpeechService with mocked paths
    tts_service = TextToSpeechService('mock_model_path', 'mock_vocoder_path')
    tts_service.synthesizer.tts = AsyncMock(return_value="Test audio output")

    # Collecting audio chunks streamed by the synthesize_stream method
    audio_chunks = []
    async for chunk in tts_service.synthesize_stream("Testing streaming", voice='default', language='en'):
        audio_chunks.append(chunk)

    # Check if the chunks are received as expected
    assert len(audio_chunks) == 5
    for i, chunk in enumerate(audio_chunks):
        assert chunk == f"Chunk {i+1} of audio for 'Testing streaming'"