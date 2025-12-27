from typing import Any, Dict
import os


class BaseGeminiClient:
    """Interface for Gemini client implementations."""

    def analyze_image(self, image) -> Dict[str, Any]:
        raise NotImplementedError

    def generate_text(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


class MockGeminiClient(BaseGeminiClient):
    """Mock client for tests and local development."""

    def analyze_image(self, image) -> Dict[str, Any]:
        # Return deterministic, structured analysis
        return {
            "description": "A simple mock image with a red square and a blue circle.",
            "objects": [
                {"label": "red square", "confidence": 0.95},
                {"label": "blue circle", "confidence": 0.9},
            ],
            "raw": {},
        }

    def generate_text(self, prompt: str) -> Dict[str, Any]:
        return {"text": "Mock response based on prompt", "prompt": prompt}


# Optional: lightweight real client wrapper
class RealGeminiClient(BaseGeminiClient):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Set GEMINI_API_KEY to use RealGeminiClient")
        try:
            import google.generativeai as genai
            self._genai = genai
            self._genai.configure(api_key=self.api_key)
        except Exception as e:
            raise ImportError("Install google-generative-ai to use RealGeminiClient") from e

    def analyze_image(self, image) -> Dict[str, Any]:
        # Practical pattern:
        # - Convert `image` (PIL.Image or path) to bytes
        # - Call the Gemini multimodal API with an image input and a text prompt asking for analysis
        # - Parse the response into a structured dict (description, objects, etc.)
        # Below is a safe template. Replace the pseudo-code with the exact SDK calls from your installed
        # google-generative-ai version.
        try:
            # Example conversion to bytes:
            from io import BytesIO
            buf = BytesIO()
            if hasattr(image, "save"):
                image.save(buf, format="PNG")
                img_bytes = buf.getvalue()
            else:
                # assume bytes already
                img_bytes = image

            # TODO: replace with actual gemini call, e.g.:
            # response = self._genai.generate(model="gemini-1.5-pro", multimodal_input={"image": img_bytes, "prompt": "Describe the image"})
            # Parse `response` into a dict below.
            raise NotImplementedError("Implement using your installed google-generative-ai SDK and model call pattern")
        except Exception as exc:
            raise NotImplementedError("RealGeminiClient.analyze_image is not implemented") from exc

    def generate_text(self, prompt: str) -> Dict[str, Any]:
        # Safe wrapper around text generation; adapt to your client's method names
        try:
            response = self._genai.generate_text(model="gemini-1.5-pro", prompt=prompt)
            # Example response access; adjust to SDK's actual response object
            return {"text": getattr(response, "text", str(response))}
        except Exception as exc:
            raise RuntimeError("Text generation failed") from exc
