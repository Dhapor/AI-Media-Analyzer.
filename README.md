
# AI Media Analyzer

Goal-driven, multimodal media analysis agent using AI (Google Gemini + Whisper + document parsing).

---

## Quick Start

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
````

2. Set the Gemini API key (if using a real Gemini client):

```bash
export GENAI_API_KEY=your_api_key_here  # Mac/Linux
set GENAI_API_KEY=your_api_key_here     # Windows
```

> Note: The project can run with mock inputs or local agent loops for testing without contacting a live API.

3. Run the Streamlit demo:

```bash
streamlit run streamlit_app/app.py
```

---

## Testing

* Install test dependencies:

```bash
pip install -r requirements.txt
```

> Using a virtual environment is recommended to avoid installing packages globally.

---

## Project Structure

* `streamlit_app/`: Main Streamlit interface for uploading media and showing summaries.
* `agent/`: Core agent logic (planning, execution, summarization).
* `google/`: Gemini client wrapper for interacting with AI APIs.
* `tests/`: Unit tests using mock Gemini clients to simulate AI responses.

---

## Design Decisions

* Keep modules small, testable, and orchestrated in `Agent`.
* Handle multiple media types: **Image, Video, Audio, PDF, Word, TXT, or Text**.
* Use **Whisper** for audio transcription and **PyPDF2 / python-docx** for document parsing.
* Structured and readable summaries over chat-style outputs.
* Display previews of images, first-frame of videos, or text snippets alongside AI summaries.

---

## Integrating a Real Gemini Client

* Implement `GeminiClient` methods to handle images, audio, and text.
* Convert PIL images, video frames, or audio to suitable formats for the AI client.
* Parse structured fields (summary, analysis, confidence) and return a consistent dictionary for the agent.
* Keep mock clients for testing and local development.

---

## Features

* Upload and analyze **Image, Video, Audio, Text, or Document** files.
* Preview images, first-frame videos, and text snippets.
* Automatic **summarization and analysis** of media content.
* Clean and intuitive UI with results displayed in a single column.
```
