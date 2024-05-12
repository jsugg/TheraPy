# app/stt/speech_to_text_service.py
import asyncio
import logging
import time
import numpy as np
from numpy.typing import NDArray

import torch
from transformers import WhisperModel, WhisperTokenizer
from transformers.tokenization_utils_base import BatchEncoding
import webrtcvad
from typing import Any, AsyncGenerator, Optional
from app.types import SingletonMeta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(name=__name__)


class SpeechToTextService(metaclass=SingletonMeta):
    def __init__(
        self,
        model_path: str,
        language_code: str = "es",
        vad_aggressiveness: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        logger.info(msg="Initializing SpeechToTextService...")
        start_time: float = time.time()
        self.language_code: str = language_code
        self.device: str = device
        self.vad_aggressiveness: int = vad_aggressiveness
        self.model_path: str = model_path

        logger.info(msg="Loading Whisper model...")
        self.model: WhisperModel | Any = WhisperModel.from_pretrained(
            pretrained_model_name_or_path=model_path
        ).to(self.device)
        self.tokenizer: WhisperTokenizer = WhisperTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path
        )
        logger.info(
            msg=f"Whisper model loaded in {time.time() - start_time:.2f} seconds on {self.device} with VAD aggressiveness level {vad_aggressiveness}."
        )
        self.vad: webrtcvad.Vad = webrtcvad.Vad(mode=vad_aggressiveness)
        end_time: float = time.time()
        logger.info(
            msg=f"SpeechToTextService initialized in {end_time - start_time:.2f} seconds."
        )

    async def transcribe_until_pause(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        sample_rate: int = 16000,
        timeout: int = 3,
    ) -> Optional[str]:
        # VAD parameters
        frame_duration: int = 3000  # Frame duration in ms (deefault: 30)
        frame_size: int = int(sample_rate * frame_duration / 1000)
        vad_buffer: NDArray[np.int16] = np.array(object=[], dtype=np.int16)
        is_speech: bool = False
        speech_detected = False
        last_speech: float = asyncio.get_event_loop().time()

        accumulated_transcription: str = ""

        try:
            start_time: float = time.time()
            logger.info(msg="Starting transcription process.")
            async for audio_chunk in audio_stream:
                vad_buffer = np.append(
                    arr=vad_buffer,
                    values=np.frombuffer(buffer=audio_chunk, dtype=np.int16),
                )

                while len(vad_buffer) >= frame_size:
                    frame: NDArray[np.int16] = vad_buffer[:frame_size]
                    vad_buffer = vad_buffer[frame_size:]
                    is_speech = self.vad.is_speech(
                        buf=frame.tobytes(), sample_rate=sample_rate
                    )
                    if is_speech:
                        logger.info(msg="Speech detected, transcribing...")
                        last_speech = asyncio.get_event_loop().time()
                        speech_detected = True
                        inputs: BatchEncoding = self.tokenizer(
                            frame, return_tensors="pt", padding=True
                        )
                        with torch.no_grad():
                            logits = self.model(
                                inputs.input_values.to(self.device)
                            ).logits
                        transcription: str = self.tokenizer.batch_decode(
                            sequences=logits.argmax(dim=-1)
                        )[0]
                        accumulated_transcription += f"{transcription} "
                        logger.info(msg=f"Transcribed: {transcription}")
                    elif speech_detected and (
                        asyncio.get_event_loop().time() - last_speech > timeout
                    ):
                        # Assume pause if no speech detected for the timeout duration
                        speech_detected = False
                        logger.info(
                            msg=f"Assuming pause. No speech detected for {timeout} seconds."
                        )
                        break
                    else:
                        # if 1 second passed and no speech detected
                        if (
                            asyncio.get_event_loop().time() - last_speech > 1
                            and not speech_detected
                        ):
                            logger.info(msg="No speech detected. Listening...")
                if not is_speech and accumulated_transcription:
                    break  # Pause detected, break the loop
            end_time: float = time.time()
            logger.info(
                msg=f"Transcription completed in {end_time - start_time:.2f} seconds."
            )
            return accumulated_transcription.strip()
        except Exception as e:
            logger.error(msg=f"Error during transcription: {e}")
            return None


# app/stt/speech_to_text_service.py

# import logging
# from typing import Any, AsyncGenerator, Literal

# import torch
# from transformers.pipelines.base import Pipeline

# from app.types import SingletonMeta
# from modules.whisper_streaming.whisper_online import (
#     OnlineASRProcessor,
#     WhisperPipelineASR,
# )

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )
# logger: logging.Logger = logging.getLogger(name=__name__)


# class SpeechToTextService(metaclass=SingletonMeta):
#     def __init__(self, language_code: str = "es") -> None:
#         device: str = ""
#         self.model: Pipeline

#         try:
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             self.asr: WhisperPipelineASR = WhisperPipelineASR(
#                 lan=language_code, torch_dtype=torch.int8
#             )
#             self.online: OnlineASRProcessor = OnlineASRProcessor(asr=self.asr)
#             logger.info(msg=f"Model loaded and ready on {device}")
#         except Exception as e:
#             logger.error(
#                 msg=f"Failed to initialize ASR service on {device}: {e}"
#             )
#             raise RuntimeError(f"ASR initialization failed: {e}") from e

#     async def transcribe_stream(
#         self, audio_stream
#     ) -> AsyncGenerator[Any | None, Any | None]:
#         """Transcribe audio in real-time from an asyncio stream."""
#         try:
#             async for audio_chunk in audio_stream:
#                 self.online.insert_audio_chunk(audio=audio_chunk)
#                 if transcription := next(self.online.process_iter(), None):
#                     yield transcription
#             if final_output := self.online.finish():
#                 yield final_output
#         except Exception as e:
#             logger.error(msg=f"Streaming transcription failed: {e}")
#             yield {"text": "Error during transcription"}

#     def load_model(self) -> None:
#         """Load the Whisper model."""
#         self.model: Pipeline = self.asr.load_model()

#     def health_check(
#         self,
#     ) -> (
#         tuple[Literal[True], Literal["STT service is operational."]]
#         | tuple[Literal[False], Literal["STT service is not operational."]]
#     ):
#         try:
#             self.online.init()  # reinitialize the processor
#             self.online.insert_audio_chunk(
#                 audio=b"Hello, this is a health check."
#             )
#             _ = next(self.online.process_iter())
#             logger.info(msg="Health check passed")
#             return True, "STT service is operational."
#         except Exception as e:
#             logger.error(msg=f"Health check failed: {str(e)}")
#             return False, "STT service is not operational."
