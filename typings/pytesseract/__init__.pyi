from typing import Final, Literal, TypedDict

from PIL.Image import Image

class TesseractError(RuntimeError): ...


class _Output:
    DICT: Final[Literal["dict"]]


Output: Final[_Output]


class OCRData(TypedDict):
    text: list[str]
    block_num: list[int | str]
    par_num: list[int | str]
    line_num: list[int | str]
    conf: list[float | int | str]
    left: list[int | str]
    top: list[int | str]
    width: list[int | str]
    height: list[int | str]


def get_languages(*, config: str = ...) -> list[str]: ...
def get_tesseract_version() -> object: ...
def image_to_osd(
    image: Image,
    *,
    timeout: int = ...,
) -> str: ...
def image_to_data(
    image: Image,
    *,
    lang: str | None = ...,
    config: str = ...,
    output_type: Literal["dict"] = ...,
    timeout: int = ...,
) -> OCRData: ...
