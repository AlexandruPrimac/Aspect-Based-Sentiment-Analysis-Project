import unittest
from unittest.mock import patch, MagicMock
from src.base import AspectSentiment
from src.llm_absa import OllamaABSA


class TestOllamaABSA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialize analyzer without actually calling Ollama server."""
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            cls.analyzer = OllamaABSA()

    # -------------------------------------------------------------
    # 1. Initialization & connectivity
    # -------------------------------------------------------------
    @patch("requests.get")
    def test_connection_check(self, mock_get):
        """Ensure that _check_ollama_connection validates the server."""
        mock_get.return_value.status_code = 200
        self.analyzer._check_ollama_connection()
        mock_get.assert_called_once()

    # -------------------------------------------------------------
    # 2. Successful JSON response parsing
    # -------------------------------------------------------------
    @patch("requests.post")
    def test_valid_response_parsing(self, mock_post):
        """Test if the model parses valid JSON correctly."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "message": {
                "content": """{
                    "aspects": [
                        {"aspect": "pizza", "sentiment": "positive", "confidence": 0.95},
                        {"aspect": "service", "sentiment": "negative", "confidence": 0.9}
                    ]
                }"""
            }
        }

        results = self.analyzer.analyze("The pizza was delicious but the service was terrible.")
        self.assertEqual(len(results), 2)
        self.assertTrue(any(r.aspect == "pizza" and r.sentiment == "positive" for r in results))
        self.assertTrue(any(r.aspect == "service" and r.sentiment == "negative" for r in results))
        self.assertIsInstance(results[0], AspectSentiment)

    # -------------------------------------------------------------
    # 3. Handle malformed JSON using regex fallback
    # -------------------------------------------------------------
    @patch("requests.post")
    def test_malformed_json_recovery(self, mock_post):
        """Ensure analyzer can recover from slightly malformed model output."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "message": {
                "content": "Here is your result: {\"aspects\": [{\"aspect\": \"food\", \"sentiment\": \"positive\", \"confidence\": 0.9}]}"
            }
        }

        results = self.analyzer.analyze("The food was great!")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].aspect, "food")
        self.assertEqual(results[0].sentiment, "positive")

    # -------------------------------------------------------------
    # 4. Empty or missing aspects
    # -------------------------------------------------------------
    @patch("requests.post")
    def test_empty_response(self, mock_post):
        """If model returns empty content, analyzer should return []"""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"message": {"content": ""}}

        results = self.analyzer.analyze("The service was slow.")
        self.assertEqual(results, [])

    # -------------------------------------------------------------
    # 5. Retry mechanism on timeout
    # -------------------------------------------------------------
    @patch("requests.post")
    def test_retry_on_timeout(self, mock_post):
        """Ensure that timeouts trigger retry logic without crashing."""
        mock_post.side_effect = [Exception("Timeout"), MagicMock(status_code=200, json=lambda: {
            "message": {"content": """{"aspects":[{"aspect":"battery","sentiment":"negative","confidence":0.8}]}"""}
        })]

        results = self.analyzer.analyze("The battery drains too fast.")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].aspect, "battery")
        self.assertEqual(results[0].sentiment, "negative")

    # -------------------------------------------------------------
    # 6. Malformed JSON that can't be fixed
    # -------------------------------------------------------------
    @patch("requests.post")
    def test_unrecoverable_json(self, mock_post):
        """If JSON parsing completely fails, return empty list."""
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "message": {"content": "This is not JSON at all."}
        }

        results = self.analyzer.analyze("The coffee was nice.")
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
