# app/tts/text_to_speech_service.py

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Generator

import torch
from transformers.utils import logging as transformers_logging
# from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.synthesizer import Synthesizer

from app.types import SingletonMeta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(name=__name__)

transformers_logging.set_verbosity_error()

class TextToSpeechService(metaclass=SingletonMeta):
    gpt_cond_latent: torch.Tensor | Any
    speaker_embedding: torch.Tensor | Any | None

    def __init__(self, config_path, model_path, language="es") -> None:
        logger.info(msg="Initializing TextToSpeechService...")
        logger.info(msg=f"Language: {language}")
        self.language: str = language
        self.voice_sample_audio_path: str = (
            "models/tts/samples/jp-en-16kHz.wav"
        )

        logger.info(msg="Loading TTS model...")
        start_time: float = time.time()
        config = XttsConfig()
        config.load_json(file_name=config_path)
        self.model: Xtts = Xtts.init_from_config(config=config)
        self.model.load_checkpoint(config=config, checkpoint_dir=model_path)
        logger.info(
            msg=f"TTS model loaded in {time.time() - start_time:.2f} seconds."
        )

        logger.info(msg="Computing speaker latents...")
        self.gpt_cond_latent, self.speaker_embedding = (
            self.model.get_conditioning_latents(
                audio_path=[self.voice_sample_audio_path]
            )
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

    async def synthesize_text(self, text: str) -> AsyncGenerator[Any, None]:
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
        return self._stream_audio(audio_chunk=audio_chunks)

    async def _stream_audio(self, audio_chunk) -> AsyncGenerator[Any, None]:
        logger.info(msg="Streaming audio...")
        buffer_size = 16000
        start_time: float = time.time()
        # for i, chunk in enumerate(iterable=audio_chunks):
        for i in range(0, len(audio_chunk), buffer_size):
            if i == 0:
                logger.info(
                    msg=f"Time to first chunck: {time.time() - start_time}"
                )
            logger.info(
                # msg=f"Received chunk {i} of audio length {chunk.shape[-1]}"
                msg=f"Received chunk {i} of audio length {len(audio_chunk)}"
            )
            # yield chunk
            # logger.info(msg=f"Audio chunk {i} streamed.")
            yield audio_chunk[i : i + buffer_size]
            logger.info(msg=f"Audio chunk streamed: {i + buffer_size}")
            await asyncio.sleep(
                0.01
            )  # Adding a short sleep to prevent overwhelming the client
