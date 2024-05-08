import pytest
import asyncio
from unittest.mock import MagicMock
from text_to_speech_service_async import TextToSpeechService

@pytest.mark.asyncio
async def test_single_synthesize():
    # Create an instance of the TextToSpeechService with mocked paths
    tts_service = TextToSpeechService('mock_model_path', 'mock_vocoder_path')
    tts_service.synthesizer.tts = MagicMock(return_value="Test audio output")
    
    # Test single asynchronous call
    response = await tts_service.synthesize("Hello world", voice='default', language='en')
    assert response == "Test audio output"

@pytest.mark.asyncio
async def test_multiple_synthesize():
    # Setup multiple concurrent calls
    tts_service = TextToSpeechService('mock_model_path', 'mock_vocoder_path')
    tts_service.synthesizer.tts = MagicMock(return_value="Test audio output")
    
    tasks = [
        asyncio.create_task(tts_service.synthesize('Hello, how are you?', voice='male', language='en')),
        asyncio.create_task(tts_service.synthesize('Good morning, how can I assist?', voice='female', language='en'))
    ]
    responses = await asyncio.gather(*tasks)
    
    # Ensure all responses are as expected
    for response in responses:
        assert response == "Test audio output"