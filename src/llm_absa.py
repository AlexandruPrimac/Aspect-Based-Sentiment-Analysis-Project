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
        self.model_name = model_name
        self.base_url = host.rstrip("/")
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

        # Validate Ollama is running
        self._check_ollama_connection()

    def _check_ollama_connection(self):
        """Check if Ollama server is accessible"""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            print(f" Connected to Ollama at {self.base_url}")
        except requests.exceptions.RequestException as e:
            print(f" Cannot connect to Ollama at {self.base_url}")
            print(f"   Error: {e}")
            print(f"   Make sure Ollama is running: ollama serve")
            raise ConnectionError(f"Ollama not available at {self.base_url}")

    def analyze(self, text: str):
        """
        Sends the text to a local Ollama LLM and returns a list of AspectSentiment objects.
        """
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

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self.temperature
            }
        }

        content = ""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                print(f" Attempt {attempt + 1}/{self.max_retries}...")
                r = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout
                )
                r.raise_for_status()

                data = r.json()
                content = data.get("message", {}).get("content", "")

                if content:
                    print(f" Got response from model")
                    break
                else:
                    print(f"  Empty response from model")

            except requests.exceptions.Timeout:
                last_error = f"Timeout after {self.timeout}s"
                print(f"  ⏱️  Timeout on attempt {attempt + 1}")
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                print(f" Request failed: {e}")
                time.sleep(1)

            except Exception as e:
                last_error = str(e)
                print(f" Unexpected error: {e}")
                time.sleep(1)

        if not content:
            print(f" Failed to get response after {self.max_retries} attempts: {last_error}")
            return []

        # Parse JSON response
        try:
            print(f" Raw content: {content[:200]}...")
            result = json.loads(content)
        except json.JSONDecodeError as e:
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

        aspects = result.get("aspects", [])
        print(f"  Found {len(aspects)} aspects")

        parsed = []
        for a in aspects:
            if "aspect" in a and "sentiment" in a:
                # Confidence score (use 1.0 for LLM-based predictions or extract if provided)
                confidence = a.get("confidence", 1.0)
                parsed.append(AspectSentiment(a["aspect"], a["sentiment"], confidence))
            else:
                print(f" Skipping malformed aspect: {a}")

        return parsed
