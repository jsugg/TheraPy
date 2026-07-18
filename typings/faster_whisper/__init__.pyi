from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

class Segment:
    text: str
    no_speech_prob: float
    avg_logprob: float
    compression_ratio: float


class TranscriptionInfo:
    language: str | None
    language_probability: float


class WhisperModel:
    def transcribe(
        self,
        audio: NDArray[np.float32],
        *,
        language: str | None = ...,
        condition_on_previous_text: bool = ...,
    ) -> tuple[Iterable[Segment], TranscriptionInfo]: ...
