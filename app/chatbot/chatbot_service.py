import logging
from typing import Any, List, Union

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from app.types import SingletonMeta
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger: logging.Logger = logging.getLogger(name=__name__)


class ChatbotService(metaclass=SingletonMeta):
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct") -> None:
        logger.info(
            msg="Initializing ChatbotService..."
        )
        start_time: float = time.time()
        try:
            logger.info(
                msg="Loading chatbot model from HuggingFace..."
            )
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
            logger.info(msg=f"Chatbot model loaded in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            logger.error(msg=f"Failed to load chatbot model: {e}")
            raise RuntimeError(f"Chatbot model loading failed: {e}") from e
        logger.info(
            msg=f"ChatbotService initialized in {time.time() - start_time:.2f} seconds."
        )

    def generate_response(self, input_text) -> str:
        start_time: float = time.time()
        try:
            inputs: List[int] = self.tokenizer.encode(
                text=input_text, return_tensors="pt"
            )
            logger.info(msg="Input text encoded successfully.")
            response: Any = self.model.generate(
                inputs, max_length=50, num_return_sequences=1
            )
            response_text: str = self.tokenizer.decode(
                token_ids=response[0], skip_special_tokens=True
            )
            logger.info(msg="Response generated successfully.")
            response_text = self.tokenizer.decode(
                token_ids=response[0], skip_special_tokens=True
            )
            logger.info(
                msg=(
                    "Response decoded successfully. Total time: "
                    f"{time.time() - start_time:.2f} seconds."
                )
            )
            return response_text
        except Exception as e:
            logger.error(msg=f"Error generating response: {e}")
            return "I'm sorry, I couldn't process your request."


# Example usage
if __name__ == "__main__":
    chatbot_service = ChatbotService()
    answer: str = chatbot_service.generate_response(
        "What is occupational therapy?"
    )
    print(answer)
