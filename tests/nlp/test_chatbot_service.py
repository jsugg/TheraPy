import pytest
from app.chatbot.chatbot_service import ChatbotService


@pytest.fixture
def chatbot():
    # Assuming the model is loaded correctly
    return ChatbotService("models/nlp/Phi-3-mini-128k-instruct-Q2_K.gguf")


def test_chatbot_valid_input(chatbot):
    # Test with a typical user query
    input_text = "Tell me about occupational therapy."
    response = chatbot.generate_response(input_text)
    assert (
        "occupational therapy" in response
    ), "The response should mention 'occupational therapy'"


def test_chatbot_edge_case_input(chatbot):
    # Test how the service handles an unexpected input
    input_text = ""
    response = chatbot.generate_response(input_text)
    assert (
        response == "I'm sorry, I couldn't process your request."
    ), "The service should handle empty inputs gracefully"


def test_chatbot_error_handling(chatbot):
    # Simulate an error in response generation
    original_generate = chatbot.generate_response

    def simulate_error(input_text):
        raise Exception("Simulated error")

    chatbot.generate_response = simulate_error
    response = chatbot.generate_response("This should fail.")
    assert (
        "I'm sorry" in response
    ), "The service should handle errors gracefully"
    chatbot.generate_response = original_generate
