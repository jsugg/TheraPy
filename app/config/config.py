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
    STT_MODEL_PATH = os.path.join(
            MODELS_DIR,
            os.getenv("STT_MODEL_PATH")
        )
    STT_MODEL_FILEPATH = os.path.join(
            MODELS_DIR,
            "stt",
            "rhasspy",
            "faster-whisper-medium-int8",
            os.environ.get("STT_MODEL_FILENAME", "model.bin")
        )
    STT_CONFIG_PATH = os.path.join(
            MODELS_DIR,
            "stt",
            "rhasspy",
            "faster-whisper-medium-int8",
            os.environ.get("STT_CONFIG_FILENAME", "config.json")
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
