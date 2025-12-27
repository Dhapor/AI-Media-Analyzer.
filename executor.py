# executor.py
import os
import tempfile
from PIL import Image
from google.genai import types

class Executor:
    """Executes actions in the Agent loop using the real Gemini API."""

    def __init__(self, gemini_client):
        self.client = gemini_client

    def execute(self, step: dict):
        action_type = step.get("action")
        params = step.get("params", {})

        if action_type == "describe_image":
            return self._describe_image(params.get("image"))
        elif action_type == "synthesize":
            return self._synthesize(params)
        elif action_type == "suggest_improvements":
            return {"text": "No suggestions available at this time."}
        else:
            return {"error": f"Unknown action: {action_type}"}

    def _describe_image(self, image: Image.Image):
        """Send image bytes to Gemini and get a text description."""

        if image is None:
            return {"error": "No image provided"}

        # Save PIL image temporarily
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, format="PNG")
            tmp_filepath = tmp.name

        # Read bytes and create a Part
        with open(tmp_filepath, "rb") as f:
            image_bytes = f.read()

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png"
        )

        # The prompt text that asks Gemini about the image
        prompt_text = "Describe the contents of this image."

        # Call generate_content with image + prompt
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image_part, prompt_text],
        )

        # Clean up the temp file
        try:
            os.remove(tmp_filepath)
        except OSError:
            pass

        # Return text response
        return {"description": response.text}

    def _synthesize(self, params: dict):
        """Use Gemini to synthesize context + goal into text."""
        context = params.get("context", {})
        goal = params.get("goal", "")

        prompt = f"Goal: {goal}\nContext: {context}\nProvide a structured summary."

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return {"text": response.text}
