# app/tts/text_to_speech_service.py

import logging
import time
from typing import Any, AsyncGenerator, Generator
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
# from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer
from app.types import SingletonMeta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TextToSpeechService(metaclass=SingletonMeta):
    gpt_cond_latent: torch.Tensor | Any
    speaker_embedding: torch.Tensor | Any | None
    # device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, config_path, model_path, language="es") -> None:
        logger.info(msg="Initializing TextToSpeechService...")
        logger.info(msg=f"Language: {language}")
        self.language: str = language
        self.voice_sample_audio_path: str = "models/tts/samples/jp-en-16kHz.wav"

        logger.info(msg="Loading TTS model...")
        start_time: float = time.time()
        config = XttsConfig()
        config.load_json(file_name=config_path)
        self.model: Xtts = Xtts.init_from_config(config=config)
        self.model.load_checkpoint(
            config=config, checkpoint_dir=model_path
        )
        logger.info(
            msg=f"TTS model loaded in {time.time() - start_time:.2f} seconds."
        )

        logger.info(msg="Computing speaker latents...")
        self.gpt_cond_latent, self.speaker_embedding = (
            self.model.get_conditioning_latents(audio_path=[self.voice_sample_audio_path])
        )

        logger.info(msg="Loading TTS synthesizer...")
        self.synthesizer = Synthesizer(
            tts_config_path=config_path,
            tts_checkpoint=model_path,
            use_cuda=torch.cuda.is_available(),
        )
        end_time: float = time.time()
        logger.info(msg="TTS synthesizer loaded successfully.")
        logger.info(
            msg=(
                "TextToSpeechService initialized in "
                f"{end_time - start_time:.2f} seconds."
            )
        )

    # async def synthesize_text(self, text_stream) -> AsyncGenerator[Any, None]:
    #     logger.info(msg=f"Starting text synthesis in language: {self.language}, for text: {text_stream}")
    #     audio_chunk, _, _ = self.synthesizer.tts(text=text_stream, language_name=self.language)
    #     logger.info(msg="Text synthesis completed.")
    #     return self._stream_audio(audio_chunk=audio_chunk)

    async def synthesize_text(self, text) -> AsyncGenerator[Any, None]:
        logger.info(
            msg=(
                f"Starting text synthesis in language: {self.language}, "
                f"for text: {text}"
            )
        )
        start_time: float = time.time()
        # audio_chunk, _, _ = self.synthesizer.tts(text=text_stream, language_name=self.language)
        audio_chunks: Generator[Any, None, None] = self.model.inference_stream(
            text=text,
            language=self.language,
            speaker_embedding=self.speaker_embedding,
            gpt_cond_latent=self.gpt_cond_latent,
            enable_text_splitting=True,
        )
        logger.info(msg="Text synthesis completed.")
        end_time: float = time.time()
        logger.info(
            msg=f"Text synthesis took {end_time - start_time:.2f} seconds."
        )

        logger.info(msg="Starting audio streaming...")
        return self._stream_audio(audio_chunks=audio_chunks)

    async def _stream_audio(self, audio_chunks) -> AsyncGenerator[Any, None]:
        logger.info(msg="Streaming audio...")
        start_time: float = time.time()
        for i, chunk in enumerate(iterable=audio_chunks):
            if i == 0:
                logger.info(
                    msg=f"Time to first chunck: {time.time() - start_time}"
                )
            logger.info(
                msg=f"Received chunk {i} of audio length {chunk.shape[-1]}"
            )
            yield chunk
            logger.info(msg=f"Audio chunk {i} streamed.")

    # async def _stream_audio(self, audio_chunk) -> AsyncGenerator[Any, None]:
    #     logger.info(msg="Streaming audio...")
    #     buffer_size = 16000  # Size of each audio chunk
    #     for i in range(0, len(audio_chunk), buffer_size):
    #         yield audio_chunk[i : i + buffer_size]
    #         logger.info(msg=f"Audio chunk streamed: {i + buffer_size}")


# import logging
# from typing import Generator
# from TTS.config import load_config
# from TTS.utils.synthesizer import Synthesizer
# from app.types import SingletonMeta

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )
# logger = logging.getLogger(__name__)


# class TextToSpeechService(metaclass=SingletonMeta):
#     def __init__(self, config_path, model_path, vocoder_path):
#         self.config = load_config(config_path)
#         try:
#             self.synthesizer = Synthesizer(
#                 tts_checkpoint=model_path,
#                 tts_config_path=config_path,
#                 vocoder_checkpoint=vocoder_path,
#                 vocoder_config=self.config,
#                 use_cuda=True,
#             )
#             logger.info("Text-to-Speech model loaded successfully.")
#         except Exception as e:
#             logger.error(f"Failed to load TTS model: {e}")
#             raise RuntimeError(
#                 f"Text-to-Speech model loading failed: {e}"
#             ) from e

#     async def synthesize_text_stream(self, text):
#         """Synthesize speech from text and stream audio chunks."""
#         audio, _, _ = self.synthesizer.tts(text=text)
#         audio_stream = self._stream_audio(audio=audio)
#         async for chunk in audio_stream:
#             yield chunk
#         logger.info("Audio synthesis streaming completed.")

#     def _stream_audio(self, audio) -> Generator[Any, Any, None]:
#         """Stream audio in chunks."""
#         buffer_size = 16000  # Define the buffer size for each audio chunk
#         for i in range(0, len(audio), buffer_size):
#             yield audio[i : i + buffer_size]

#     def synthesize_text(self, text):
#         try:
#             audio, _, _ = self.synthesizer.tts(text)
#             logger.info("Audio synthesis successful.")
#             return audio
#         except Exception as e:
#             logger.error(f"Error during text synthesis: {e}")
#             return None


# # Example usage
# if __name__ == "__main__":
#     tts_service = TextToSpeechService(
#         config_path="config.json",
#         model_path="tts_model.pth.tar",
#         vocoder_path="vocoder_model.pth.tar",
#     )
#     audio_output = tts_service.synthesize_text(
#         "Hello, how can I assist you today?"
#     )
