"""
Implementation 3: LLM-Based ABSA using a locally running model through Ollama.

This approach delegates both aspect extraction and sentiment classification
to a local Large Language Model. The model is accessed via Ollama's REST API.

I am using 'deepseek-v3.1:671b-cloud' because it has a very nice efficiency and accuracy when it comes to predicting the sentiments.
I tried using Mistral, Gemma3:1b but these models are small and the prompt engineering part is not going to help me that much.

Key ideas:
- Use prompt engineering to request structured JSON output
- Handle request retries, timeouts, and malformed JSON gracefully
- Convert the model's JSON output into AspectSentiment objects

This method is slower but often more flexible and context-aware than
rule-based or transformer approaches.
"""

from __future__ import annotations
import json
import re
import time
import requests
from src.base import ABSAAnalyzer, AspectSentiment


class OllamaABSA(ABSAAnalyzer):
    """
    Simple ABSA via a local LLM served by Ollama.
    """

    def __init__(
            self,
            model_name: str = "deepseek-v3.1:671b-cloud",
            host: str = "http://localhost:11434",
            temperature: float = 0.2,
            max_retries: int = 2,
            timeout: int = 30,
    ):
        # Store model configuration
        self.model_name = model_name
        self.base_url = host.rstrip("/")
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

        # Check Ollama connection on startup to avoid silent failures later
        self._check_ollama_connection()

    def _check_ollama_connection(self):
        """Check if Ollama server is accessible"""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            print(f" Connected to Ollama at {self.base_url}")

        except requests.exceptions.RequestException as e:
            # Provide clear feedback if Ollama isn’t running
            print(f" Cannot connect to Ollama at {self.base_url}")
            print(f"   Error: {e}")
            print(f"   Make sure Ollama is running: ollama serve")
            raise ConnectionError(f"Ollama not available at {self.base_url}")

    def analyze(self, text: str):
        """
        Analyze the given text using a local LLM.

        Steps:
        1. Build a structured prompt using the techniques learned in class, instructing the model to return valid JSON.
        2. Send request via Ollama’s REST API.
        3. Retry on failure (timeouts, network issues).
        4. Parse the model’s JSON response safely.
        5. Return a list of AspectSentiment objects.
        """

        # Prompt design: very explicit JSON-only instruction for consistency
        prompt = f"""You are an aspect-based sentiment analyzer.
        Extract every aspect mentioned in the text and classify its sentiment as positive, negative, or neutral.
        Also provide a confidence score between 0 and 1.

        Return ONLY valid JSON like this:
        {{
            "aspects": [
        {{"aspect": "pizza", "sentiment": "positive", "confidence": 0.95}},
        {{"aspect": "service", "sentiment": "negative", "confidence": 0.90}}
        ]
        }}

        Text: "{text}"
        """

        # Build the JSON payload for the Ollama API
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json", # Enforce JSON output
            "stream": False,  # Disable streaming for easier handling
            "options": {
                "temperature": self.temperature
            }
        }

        content = ""  # Will hold the raw model response
        last_error = None # Track the last error for reporting

        # Retry mechanism, useful if the model or API occasionally fails
        for attempt in range(self.max_retries):
            try:
                print(f" Attempt {attempt + 1}/{self.max_retries}...")
                r = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout
                )
                r.raise_for_status()

                # Extract content field from Ollama’s response JSON
                data = r.json()
                content = data.get("message", {}).get("content", "")

                if content:
                    print(f" Got response from model")
                    break
                else:
                    print(f" Empty response from model")

            # Handle network or timeout errors gracefully
            except requests.exceptions.Timeout:
                last_error = f"Timeout after {self.timeout}s"
                print(f" Timeout on attempt {attempt + 1}")
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                print(f" Request failed: {e}")
                time.sleep(1)

            except Exception as e:
                last_error = str(e)
                print(f" Unexpected error: {e}")
                time.sleep(1)

        # If all attempts fail, stop execution early
        if not content:
            print(f" Failed to get response after {self.max_retries} attempts: {last_error}")
            return []

        # ------------------------------------------------------------------ #
        # Step 2: Try to parse the JSON output from the model
        # ------------------------------------------------------------------ #
        try:
            print(f" Raw content: {content[:200]}...")
            result = json.loads(content)
        except json.JSONDecodeError as e:
            # Handle malformed responses (sometimes the model adds extra text)
            print(f" JSON decode failed: {e}")
            print(f" Trying regex extraction...")
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(0))
                    print(f"  Extracted JSON via regex")
                except json.JSONDecodeError:
                    print(f"  Regex extraction also failed")
                    return []
            else:
                print(f"  No JSON found in response")
                return []

        # Extract aspect list from parsed JSON
        aspects = result.get("aspects", [])
        print(f"  Found {len(aspects)} aspects")

        parsed = []
        # Convert each JSON object into an AspectSentiment instance
        for a in aspects:
            if "aspect" in a and "sentiment" in a:
                # Use provided confidence if available; otherwise default to 1.0
                confidence = a.get("confidence", 1.0)
                parsed.append(AspectSentiment(a["aspect"], a["sentiment"], confidence))
            else:
                print(f" Skipping malformed aspect: {a}")

        # Final structured output, ready for evaluation or display
        return parsed
