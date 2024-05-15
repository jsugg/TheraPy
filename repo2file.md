# Project Codebase
## Root directory
 `/Users/juanpedrosugg/dev/github/otherapy`
---
## Directory structure:
```
.
├── Dockerfile
├── LICENSE
├── Pipfile
├── Pipfile.lock
├── README.md
├── app
│   ├── __init__.py
│   ├── config
│   │   ├── __init__.py
│   │   └── config.py
│   ├── integration_controller.py
│   ├── services
│   │   ├── chatbot
│   │   │   ├── Dockerfile
│   │   │   ├── __init__.py
│   │   │   └── chatbot_service.py
│   │   ├── stt
│   │   │   ├── Dockerfile
│   │   │   ├── __init__.py
│   │   │   └── speech_to_text_service.py
│   │   └── tts
│   │       ├── Dockerfile
│   │       ├── __init__.py
│   │       └── text_to_speech_service.py
│   ├── types
│   │   ├── __init__.py
│   │   └── singletonmeta.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── hf-whisper-port.py
│   └── web
│       ├── Dockerfile
│       ├── __init__.py
│       ├── flask_app
│       │   ├── __init__.py
│       │   └── flask_app.py
│       ├── main.py
│       └── templates
│           └── index.html
├── config.json
├── data
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-311.pyc
│   └── database.py
├── docker-compose.yml
├── docs
├── extra_installs.sh
├── models
│   ├── nlp
│   │   ├── Phi-3-mini-128k-instruct-Q2_K.gguf
│   │   ├── README.md
│   │   ├── config.json
│   │   ├── merges.txt
│   │   ├── microsoft
│   │   │   └── Phi-3-mini-128k-instruct
│   │   │       ├── added_tokens.json
│   │   │       ├── config.json
│   │   │       ├── configuration_phi3.py
│   │   │       ├── generation_config.json
│   │   │       ├── model-00001-of-00002.safetensors
│   │   │       ├── model-00002-of-00002.safetensors
│   │   │       ├── model.safetensors.index.json
│   │   │       ├── modeling_phi3.py
│   │   │       ├── special_tokens_map.json
│   │   │       ├── tokenizer.json
│   │   │       ├── tokenizer.model
│   │   │       └── tokenizer_config.json
│   │   ├── pytorch_model.bin
│   │   ├── samples
│   │   │   ├── samples_en_sample.wav
│   │   │   └── samples_es_sample.wav
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── stt
│   │   ├── rhasspy
│   │   │   └── faster-whisper-medium-int8
│   │   │       ├── README.md
│   │   │       ├── config.json
│   │   │       ├── model.bin
│   │   │       └── vocabulary.txt
│   │   └── sanchit-gandhi
│   │       └── whisper-medium-fleurs-lang-id
│   │           ├── README.md
│   │           ├── all_results.json
│   │           ├── config.json
│   │           ├── ds_config.json
│   │           ├── eval_results.json
│   │           ├── gitattributes
│   │           ├── gitignore
│   │           ├── model.safetensors
│   │           ├── preprocessor_config.json
│   │           ├── pytorch_model.bin
│   │           ├── run.sh
│   │           ├── run_audio_classification.py
│   │           ├── train_results.json
│   │           ├── trainer_state.json
│   │           └── training_args.bin
│   └── tts
│       ├── LICENSE.txt
│       ├── README.md
│       ├── config.json
│       ├── dvae.pth
│       ├── hash.md5
│       ├── mel_stats.pth
│       ├── model.pth
│       ├── samples
│       │   ├── jp-en-16kHz.wav
│       │   └── jp-en.m4a
│       ├── speakers_xtts.pth
│       └── vocab.json
├── modules
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-311.pyc
│   └── whisper_streaming
│       ├── LICENSE
│       ├── README.md
│       ├── client-mac.sh
│       ├── client-translate.sh
│       ├── client.sh
│       ├── cuda_check.py
│       ├── line_packet.py
│       ├── mlx_whisper.py
│       ├── server-hf-v3.sh
│       ├── server-mlx-v3.sh
│       ├── server-v3.sh
│       ├── server.sh
│       ├── translate
│       │   ├── README.md
│       │   ├── translate-glm-out.txt
│       │   ├── translate-glm.py
│       │   ├── translate-mistral-out.txt
│       │   ├── translate-mistral.py
│       │   ├── translate-nllb-out.txt
│       │   ├── translate-nllb.py
│       │   ├── translate-qwen1_8b-out.txt
│       │   ├── translate-qwen1_8b.py
│       │   ├── translate-t5-marian-out.txt
│       │   ├── translate-t5-marian.py
│       │   ├── translate.py
│       │   ├── translate_benchmark.py
│       │   └── translate_source.py
│       ├── whisper_online.py
│       └── whisper_online_server.py
├── pyproject.toml
├── repo2file.md
├── requirements.txt
├── static
│   └── assets
└── tests
    ├── __init__.py
    ├── __pycache__
    │   └── __init__.cpython-311.pyc
    ├── nlp
    │   └── test_chatbot_service.py
    ├── stt
    │   └── test_speech_to_text_service.py
    └── tts
        └── test_text_to_speech_service.py

38 directories, 125 files
```

---
## File: app/services/chatbot/__init__.py
```
from .chatbot_service import ChatbotService
```

---
## File: app/services/chatbot/chatbot_service.py
```
# app/services/chatbot/chatbot_service.py
import logging
import warnings
from typing import Any, List, Union

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging as transformers_logging

from app.types import SingletonMeta
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(name=__name__)
transformers_logging.set_verbosity_error()
warnings.filterwarnings(
    action="ignore", message="`flash-attention` package not found"
)
warnings.filterwarnings(
    action="ignore",
    message="Current `flash-attenton` does not support `window_size`",
)


class ChatbotService(metaclass=SingletonMeta):
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct") -> None:
        logger.info(msg="Initializing ChatbotService...")
        start_time: float = time.time()
        try:
            logger.info(msg="Loading chatbot model from HuggingFace...")
            self.tokenizer: Union[
                PreTrainedTokenizer, PreTrainedTokenizerFast
            ] = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name,
                trust_remote_code=True,
            )
            self.model: Any = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name,
                trust_remote_code=True,
            )
            logger.info(
                msg=f"Chatbot model loaded in {time.time() - start_time:.2f} seconds."
            )
        except Exception as e:
            logger.error(msg=f"Failed to load chatbot model: {e}")
            raise RuntimeError(f"Chatbot model loading failed: {e}") from e
        logger.info(
            msg=f"ChatbotService initialized in {time.time() - start_time:.2f} seconds."
        )

    def generate_response(self, input_text) -> str:
        try:
            start_time: float = time.time()
            logger.info(
                msg=(f"Generating response for input text: {input_text}. ")
            )
            response: str = self._generate_response(prompt=input_text)
            logger.info(
                msg=(
                    "Response generated successfully. Total time: "
                    f"{time.time() - start_time:.2f} seconds."
                    f" Response: {response}"
                )
            )
            return response
        except Exception as e:
            logger.error(msg=f"Error generating response: {e}")
            return "I'm sorry, I couldn't process your request."

    def _generate_response(self, prompt) -> str:
        inputs: List[int] = self.tokenizer.encode(
            text=prompt, return_tensors="pt"
        )
        logger.info(msg="Prompt encoded successfully.")
        response: Any = self.model.generate(
            inputs, max_length=50, num_return_sequences=1
        )
        logger.info(msg="Response generated successfully.")
        response_text: str = self.tokenizer.decode(
            token_ids=response[0], skip_special_tokens=True
        )
        logger.info(msg="Response text decoded successfully.")
        return response_text


# Example usage
if __name__ == "__main__":
    chatbot_service = ChatbotService()
    answer: str = chatbot_service.generate_response(
        input_text="What is occupational therapy?"
    )
    print(answer)
```

---
## File: app/services/stt/Dockerfile
```
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies for Whisper
RUN apt-get update && apt-get install -y libsndfile1

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Run app.py when the container launches
CMD ["python", "app.py"]```

---
## File: app/services/stt/__init__.py
```
from .speech_to_text_service import SpeechToTextService

```

---
## File: app/services/stt/speech_to_text_service.py
```
# app/stt/speech_to_text_service.py
import asyncio
import logging
import warnings
import time
import numpy as np
from numpy.typing import NDArray

import torch
from transformers import WhisperModel, WhisperTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import logging as transformers_logging
import webrtcvad

from app.types import SingletonMeta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(name=__name__)
transformers_logging.set_verbosity_error()
warnings.filterwarnings(
    action="ignore",
    message="`resume_download` is deprecated and will be removed in version",
)

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
        self.model: WhisperModel = WhisperModel.from_pretrained(
            pretrained_model_name_or_path=model_path
        ).to(self.device)
        self.tokenizer: WhisperTokenizer = WhisperTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path
        )
        logger.info(
            msg=f"Whisper model loaded on {self.device} with VAD aggressiveness level {vad_aggressiveness}."
        )
        self.vad: webrtcvad.Vad = webrtcvad.Vad(mode=vad_aggressiveness)
        end_time: float = time.time()
        logger.info(
            msg=f"SpeechToTextService initialized in {end_time - start_time:.2f} seconds."
        )

    async def transcribe_until_pause(
        self,
        audio_stream,
        sample_rate: int = 16000,
        timeout: int = 3,
    ) -> str | None:
        # VAD parameters
        frame_duration: int = 30  # Frame duration in ms (default: 30)
        frame_size: int = int(sample_rate * frame_duration / 1000)
        vad_buffer: NDArray[np.int16] = np.array(object=[], dtype=np.int16)
        is_speech: bool = False
        speech_detected: bool = False
        last_speech: float = asyncio.get_event_loop().time()
        accumulated_transcription: str = ""
        frame: NDArray[np.int16]

        try:
            start_time: float = time.time()
            logger.info(msg="Starting transcription process.")
            async for audio_chunk in audio_stream:
                vad_buffer: NDArray[np.int16] = np.append(
                    arr=vad_buffer,
                    values=np.frombuffer(buffer=audio_chunk, dtype=np.int16),
                )

                while len(vad_buffer) >= frame_size:
                    frame = vad_buffer[:frame_size]
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
                            logits: torch.Tensor = self.model(
                                inputs.input_values.to(self.device)
                            ).logits
                        transcription: str = self.tokenizer.batch_decode(
                            sequences=logits.argmax(dim=-1)
                        )[0]
                        accumulated_transcription += f"{transcription} "
                        logger.debug(msg=f"Transcribed: {transcription}")
                    elif speech_detected and (
                        asyncio.get_event_loop().time() - last_speech > timeout
                    ):
                        # Assume pause if no speech detected
                        # for the timeout duration
                        speech_detected = False
                        logger.info(
                            msg=(
                                "Assuming pause. "
                                f"No speech detected for {timeout} seconds."
                            )
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
                msg=(
                    "Transcription completed in "
                    f"{end_time - start_time:.2f} seconds."
                    f" Transcription: {accumulated_transcription}"
                )
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
```

---
## File: app/services/tts/__init__.py
```
from .text_to_speech_service import TextToSpeechService
```

---
## File: app/services/tts/text_to_speech_service.py
```
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
```

---
## File: app/web/Dockerfile
```
# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app_interface.py when the container launches
CMD ["streamlit", "run", "app_interface.py"]
```

---
## File: app/web/flask_app/__init__.py
```
from .flask_app import create_app
```

---
## File: app/web/flask_app/flask_app.py
```

from flask import Flask, Blueprint
from flask_socketio import SocketIO

def create_app() -> Flask:
    app: Flask = Flask(import_name=__name__)
    app.config['SECRET_KEY'] = 'my_secret_key'
    return app
```

---
## File: app/web/main.py
```
# app/web/run.py
import asyncio

from typing import AsyncGenerator, Any, Coroutine
from flask import Flask, Blueprint, render_template
from flask_socketio import SocketIO, emit

from app.integration_controller import IntegrationController
from app.services.chatbot import ChatbotService
from app.services.stt import SpeechToTextService
from app.services.tts import TextToSpeechService
from app.web.flask_app import create_app


# Flask app initialization
main: Blueprint = Blueprint(
    name="main", import_name=__name__, template_folder="templates"
)
app: Flask = create_app()
socketio: SocketIO = SocketIO(app=app, cors_allowed_origins="*")

# Initialize ML services
stt_service = SpeechToTextService(model_path="openai/whisper-medium")
chatbot_service = ChatbotService(model_name="microsoft/Phi-3-mini-4k-instruct")
tts_service = TextToSpeechService(
    config_path="models/tts/config.json", model_path="models/tts/"
)

# Initialize integration controller
controller = IntegrationController(
    stt_service=stt_service,
    chatbot_service=chatbot_service,
    tts_service=tts_service,
)


# Define routes
@main.route(rule="/")
def index() -> str:
    return render_template(template_name_or_list="index.html")


# Define socket events
@socketio.on("stream_audio")
def handle_audio_stream(data) -> None:
    audio_stream = data["audio"]

    async def process_audio() -> None:
        async for audio_chunk in controller.handle_interaction(
            audio_data={"audio": audio_stream}
        ):
            socketio.emit("audio_response", {"audio": audio_chunk})

    asyncio.create_task(coro=process_audio())


# Register routes and socket events
app.register_blueprint(blueprint=main, url_prefix="/")
socketio.init_app(app=app)

# Run the Flask app
if __name__ == "__main__":
    socketio.run(app=app, debug=True, use_reloader=False, log_output=True)
```

---
## File: app/web/templates/index.html
```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Real-Time Voice Chatbot</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function () {
            var socket = io();
            var audioContext = new AudioContext();

            navigator.mediaDevices.getUserMedia({ audio: true, video: false })
                .then(function(stream) {
                    const mediaStreamSource = audioContext.createMediaStreamSource(stream);
                    const processor = audioContext.createScriptProcessor(1024, 1, 1);
                    mediaStreamSource.connect(processor);
                    processor.connect(audioContext.destination);

                    processor.onaudioprocess = function(e) {
                        const inputData = e.inputBuffer.getChannelData(0);
                        const inputDataArray = Array.from(inputData);
                        socket.emit('stream_audio', { audio: inputDataArray });
                    };

                    socket.on('audio_response', function(data) {
                        playAudio(data.audio);
                    });

                    console.log('Audio capture and streaming started.');
                })
                .catch(function(err) {
                    console.error('Audio capture error:', err);
                });

            function playAudio(audioData) {
                var arrayBuffer = new Float32Array(audioData).buffer;
                audioContext.decodeAudioData(arrayBuffer, function(buffer) {
                    var source = audioContext.createBufferSource();
                    source.buffer = buffer;
                    source.connect(audioContext.destination);
                    source.start(0);
                }, function(e) {
                    console.log('Error decoding audio data:', e);
                });
            }
        });
    </script>
</head>
<body>
    <h1>Real-Time Voice Chatbot</h1>
</body>
</html>
```

---
## File: Dockerfile
```
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install necessary libraries
RUN apt-get update && apt-get install -y \
    libsndfile1

# Copy the local code to the container's workspace
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]
```

---
## File: app/__init__.py
```
# app/__init__.py
import sys

from app.config import Config
from app.types import SingletonMeta

# Register globals
globals()["SingletonMeta"] = SingletonMeta
globals()["Config"] = Config

package_name: str = __name__.split(sep=".", maxsplit=1)[0]
sys.modules[package_name].__dict__["SingletonMeta"] = SingletonMeta
sys.modules[package_name].__dict__["Config"] = Config
```

---
## File: app/config/__init__.py
```
from .config import Config  # pylint: disable=unused-import
```

---
## File: app/config/config.py
```
# app/types/config.py

import os
from pathlib import Path

from dotenv import load_dotenv

from app.types import SingletonMeta

# Load environment variables
BASE_DIR = Path(__file__).parent.parent
load_dotenv(os.path.join(BASE_DIR, ".env"))


class Config(SingletonMeta):
    # Set up paths
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    # Configure paths
    PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed")

    # Coqui STT paths
    STT_MODEL_PATH: str = os.path.join(
            MODELS_DIR,
            os.environ.get("STT_MODEL_PATH", default="openai/whisper-medium")
        )
    STT_MODEL_FILEPATH = os.path.join(
            MODELS_DIR,
            "stt",
            "rhasspy",
            "faster-whisper-medium-int8",
            os.environ.get("STT_MODEL_FILENAME", default="model.bin")
        )
    STT_CONFIG_PATH = os.path.join(
            MODELS_DIR,
            "stt",
            "rhasspy",
            "faster-whisper-medium-int8",
            os.environ.get("STT_CONFIG_FILENAME", default="config.json")
        )
    # STT_SCORER_PATH = os.path.join(MODELS_DIR, "stt", "scorer.scorer")

    # Coqui TTS paths
    TTS_CONFIG_PATH = os.path.join(MODELS_DIR, "tts", "config.json")
    TTS_MODEL_PATH = os.path.join(MODELS_DIR, "tts", "tts_model.pth")

    # NLP model paths
    ROBERTA_MODEL_PATH = os.path.join(MODELS_DIR, "nlp", "roberta")
    UNSLOTH_MODEL_PATH = os.path.join(MODELS_DIR, "nlp", "unsloth")

    # Weaviate configuration
    WEAVIATE_ENDPOINT = os.getenv("WEAVIATE_ENDPOINT")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
```

---
## File: app/integration_controller.py
```
# app/integration_controller.py

import asyncio
import logging
from typing import Optional
from app.services.stt import SpeechToTextService
from app.services.tts import TextToSpeechService
from app.services.chatbot import ChatbotService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(name=__name__)


class IntegrationController:
    def __init__(self, stt_service, chatbot_service, tts_service) -> None:
        logger.info(
            msg=(
                "Initializing IntegrationController with STT, "
                "Chatbot, and TTS services."
            )
        )
        self.stt_service: SpeechToTextService = stt_service
        self.chatbot_service: ChatbotService = chatbot_service
        self.tts_service: TextToSpeechService = tts_service
        self.loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop=self.loop)
        
        logger.info(
            msg="IntegrationController initialized."
        )

    async def handle_interaction(
        self, audio_data
    ):
        logger.info(msg="Starting interaction handling process.")
        try:
            transcription: Optional[str] = (
                await self.stt_service.transcribe_until_pause(
                    audio_stream=audio_data['audio']
                )
            )
            if transcription:
                logger.info(msg=f"Transcription: {transcription}")
                response_text: str = self.chatbot_service.generate_response(
                    input_text=transcription
                )
                logger.info(msg=f"Generated response: {response_text}")
                async for audio_chunk in self.tts_service.synthesize_text(text=response_text):
                    yield audio_chunk
                # async for audio_chunk in self.tts_service.synthesize_text(
                #     text=response_text
                # ):
                #     yield audio_chunk
                #     logger.info(msg="Audio chunk synthesized and yielded.")
            else:    
                logger.info(msg="SpeechToTextService - Transcription from the audiostream is None.")
        except Exception as e:
            logging.error(msg=f"Failed to handle interaction: {e}")
            return None

    def cleanup(self) -> None:
        logger.info(msg="Cleaning up resources.")
        try:
            if self.loop.is_running():
                logger.info(msg="Stopping the event loop.")
                self.loop.stop()
                logger.info(msg="Event loop stopped.")
        except RuntimeError:
            logger.warning(msg="Event loop was already stopped.")
        finally:
            logger.info(msg="Cleanup completed.")
```

---
## File: app/types/__init__.py
```
from typing import Any, TypeVar
from .singletonmeta import SingletonMeta  # pylint: disable=unused-import 

AIModel = TypeVar("AIModel", bound=Any)```

---
## File: app/types/singletonmeta.py
```
"""
Module for creating singleton instances of classes.
"""
import threading
from typing import Any, Dict


class SingletonMeta(type):
    """Metaclass for creating singleton instances of classes.

    This metaclass ensures that only one instance of each class using
    it is created. If an instance of the class already exists, it
    returns the existing instance.

    Args:
        cls: The class to create a singleton instance of.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The singleton instance of the class.
    """

    _instances: dict = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance: Any = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]
```

---
