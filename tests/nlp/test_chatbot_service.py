import unittest
from src.chatbot.chatbot_service import ChatbotService

class TestChatbotService(unittest.TestCase):
    def test_response_generation(self):
        chatbot_service = ChatbotService()
        response = chatbot_service.generate_response("What is today's weather?")
        self.assertIn("weather", response)

if __name__ == '__main__':
    unittest.main()
