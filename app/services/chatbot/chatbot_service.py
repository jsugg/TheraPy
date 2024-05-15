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
