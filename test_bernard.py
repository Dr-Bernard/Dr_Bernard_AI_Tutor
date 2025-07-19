import unittest
from unittest.mock import patch, MagicMock
from bernard import ChatAgent

class TestChatAgent(unittest.TestCase):

    @patch('bernard.Groq')
    def test_get_response_no_search(self, MockGroq):
        # Mock the Groq client and its response
        mock_groq_instance = MockGroq.return_value
        mock_chat_completion = MagicMock()
        mock_chat_completion.choices = [MagicMock()]
        mock_chat_completion.choices[0].message.content = "This is a test response."
        mock_groq_instance.chat.completions.create.return_value = mock_chat_completion

        # Initialize the ChatAgent
        agent = ChatAgent(api_key="fake_api_key", model_name="fake_model")

        # Test the get_response method
        response = agent.get_response("Hello")
        self.assertEqual(response, "This is a test response.")
        self.assertEqual(len(agent.conversation_history), 2)
        self.assertEqual(agent.conversation_history[0]["role"], "user")
        self.assertEqual(agent.conversation_history[0]["content"], "Hello")
        self.assertEqual(agent.conversation_history[1]["role"], "assistant")
        self.assertEqual(agent.conversation_history[1]["content"], "This is a test response.")

    @patch('bernard.requests.get')
    def test_web_search(self, mock_get):
        # Mock the requests.get method and its response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "webPages": {
                "value": [
                    {
                        "snippet": "This is a web search result."
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        # Initialize the ChatAgent
        agent = ChatAgent(api_key="fake_api_key", model_name="fake_model")

        # Test the web_search method
        result = agent.web_search("test query")
        self.assertEqual(result, "This is a web search result.")

    @patch('bernard.ChatAgent.web_search')
    def test_get_response_with_search(self, mock_web_search):
        # Mock the web_search method
        mock_web_search.return_value = "This is a web search result."

        # Initialize the ChatAgent
        agent = ChatAgent(api_key="fake_api_key", model_name="fake_model")

        # Test the get_response method with a search query
        response = agent.get_response("search for something")
        self.assertEqual(response, "This is a web search result.")
        self.assertEqual(len(agent.conversation_history), 2)
        self.assertEqual(agent.conversation_history[0]["role"], "user")
        self.assertEqual(agent.conversation_history[0]["content"], "search for something")
        self.assertEqual(agent.conversation_history[1]["role"], "assistant")
        self.assertEqual(agent.conversation_history[1]["content"], "This is a web search result.")

if __name__ == '__main__':
    unittest.main()
