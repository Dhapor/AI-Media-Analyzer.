import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
from PIL import Image
import tempfile
import os
import sys
import cv2
import whisper
from PyPDF2 import PdfReader
from docx import Document
import warnings
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import Agent
from google.genai import Client as GeminiClient


st.set_page_config(page_title="AI Media Analyzer", layout="wide")
st.title("AI Media Analyzer")
# st.write("Upload an Image, Video, Audio, PDF, Word, TXT, or Text and let the AI Agent summarize and analyze it.")

MAX_WIDTH = 700
MAX_HEIGHT = 500
MAX_AGENT_DIM = 512  
MAX_TEXT_LEN = 3000  

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


col1, spacer, col2 = st.columns([1, 0.2, 2])

with col1:
    # Project Title
    # st.title("AI Media Analyzer")
    st.image("g.jpg", width=300)
    st.write("Upload an Image, Video, Audio, PDF, Word, TXT, or Text and let the AI Agent summarize and analyze it.")

    st.header("Input")

    # Simplified media type selector
    media_type = st.radio(
        "Select input type:",
        ["Image", "Video", "Audio", "Text / Document"]
    )

    if "last_media_type" not in st.session_state or st.session_state.last_media_type != media_type:
        st.session_state.agent_result = None
    st.session_state.last_media_type = media_type

    max_iters = st.slider("Max iterations", min_value=1, max_value=5, value=2)

    uploaded_file = None
    file_text = None

    # File uploader
    types_map = {
        "Image": ["png", "jpg", "jpeg"],
        "Video": ["mp4", "mov", "avi"],
        "Audio": ["mp3", "wav", "m4a"],
        "Text / Document": ["pdf", "docx", "txt"]
    }
    uploaded_file = st.file_uploader(f"Upload a {media_type} file", type=types_map[media_type])

    # Text input for Text / Document
    if media_type == "Text / Document":
        text_input = st.text_area("Or type/paste text here")
        if text_input:
            file_text = text_input

    run_agent = st.button("Run Agent")

    if run_agent:
        if (uploaded_file is None) and (file_text is None):
            st.error("Please provide an input first.")
        else:
            api_key = os.environ.get("GENAI_API_KEY")
            if not api_key:
                st.error("GENAI_API_KEY environment variable not set.")
            else:
                client = GeminiClient(api_key=api_key)
                agent = Agent(client)
                st.info("Running Agent...")

                agent_input = {}
                try:
                    
                    if media_type == "Image" and uploaded_file:
                        uploaded_file.seek(0)
                        img = Image.open(uploaded_file)
                        # Preview resizing
                        preview_img = img.copy()
                        if preview_img.width > MAX_WIDTH or preview_img.height > MAX_HEIGHT:
                            ratio = min(MAX_WIDTH / preview_img.width, MAX_HEIGHT / preview_img.height)
                            preview_img = preview_img.resize((int(preview_img.width*ratio), int(preview_img.height*ratio)), Image.Resampling.LANCZOS)
                        

                        # Agent resizing
                        agent_img = img.copy()
                        if agent_img.width > MAX_AGENT_DIM or agent_img.height > MAX_AGENT_DIM:
                            ratio = min(MAX_AGENT_DIM / agent_img.width, MAX_AGENT_DIM / agent_img.height)
                            agent_img = agent_img.resize((int(agent_img.width*ratio), int(agent_img.height*ratio)), Image.Resampling.LANCZOS)
                        agent_input["image"] = agent_img

                    
                    elif media_type == "Video" and uploaded_file:
                        uploaded_file.seek(0)
                        temp_vid = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        temp_vid.write(uploaded_file.read())
                        temp_vid.close()

                        cap = cv2.VideoCapture(temp_vid.name)
                        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                        frame = None
                        for i in range(0, total_frames, max(1, fps)):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                            ret, f = cap.read()
                            if ret:
                                frame = f
                                break
                        cap.release()
                        if frame is not None:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_frame = Image.fromarray(frame_rgb)
                            preview_frame = pil_frame.copy()
                            if preview_frame.width > MAX_WIDTH or preview_frame.height > MAX_HEIGHT:
                                ratio = min(MAX_WIDTH / preview_frame.width, MAX_HEIGHT / preview_frame.height)
                                preview_frame = preview_frame.resize((int(preview_frame.width*ratio), int(preview_frame.height*ratio)), Image.Resampling.LANCZOS)
                            st.subheader("Preview (first frame)")
                            st.image(preview_frame)

                            # Agent resizing
                            if pil_frame.width > MAX_AGENT_DIM or pil_frame.height > MAX_AGENT_DIM:
                                ratio = min(MAX_AGENT_DIM / pil_frame.width, MAX_AGENT_DIM / pil_frame.height)
                                pil_frame = pil_frame.resize((int(pil_frame.width*ratio), int(pil_frame.height*ratio)), Image.Resampling.LANCZOS)
                            agent_input["image"] = pil_frame

                        os.unlink(temp_vid.name)

                    
                    elif media_type == "Audio" and uploaded_file:
                        uploaded_file.seek(0)
                        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        temp_audio.write(uploaded_file.read())
                        temp_audio.close()
                        st.subheader("Preview")
                        st.audio(temp_audio.name)
                        agent_input["audio_path"] = temp_audio.name

                        # Transcribe with Whisper
                        model = load_whisper_model()
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            result_whisper = model.transcribe(temp_audio.name)
                        transcription = result_whisper["text"]
                        agent_input["text"] = transcription
                        os.unlink(temp_audio.name)

                    
                    elif media_type == "Text / Document":
                        if uploaded_file:
                            uploaded_file.seek(0)
                            file_ext = uploaded_file.name.split(".")[-1].lower()
                            if file_ext == "pdf":
                                reader = PdfReader(uploaded_file)
                                pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
                                file_text = "\n".join(pages_text)
                            elif file_ext == "docx":
                                doc = Document(uploaded_file)
                                paragraphs = [p.text for p in doc.paragraphs if p.text]
                                file_text = "\n".join(paragraphs)
                            elif file_ext == "txt":
                                file_text = uploaded_file.read().decode("utf-8")

                        if file_text:
                            # Limit large text for faster agent processing
                            file_text_limited = file_text[:MAX_TEXT_LEN]
                            agent_input["text"] = file_text_limited
                            st.subheader("Preview")
                            st.write(file_text[:1000] + ("..." if len(file_text) > 1000 else ""))

                    
                    if agent_input:
                        with st.spinner("Agent is analyzing your input..."):
                            st.session_state["agent_result"] = agent.run(
                                goal="Summarize and analyze input",
                                inputs=agent_input,
                                max_iters=max_iters
                            )
                        st.success("✅ Agent run completed.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")

preview_image = None

with col1:
    if media_type == "Image" and uploaded_file:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        if img.width > MAX_WIDTH or img.height > MAX_HEIGHT:
            ratio = min(MAX_WIDTH / img.width, MAX_HEIGHT / img.height)
            img = img.resize((int(img.width*ratio), int(img.height*ratio)), resample=Image.Resampling.LANCZOS)
        preview_image = img  # store the image to show later in col2


with col2:
    # Show image preview if available
    if preview_image:
        st.subheader("Preview")
        st.image(preview_image, width=600)

    if "agent_result" in st.session_state and st.session_state["agent_result"]:
        result = st.session_state["agent_result"]
        st.header("Output")
        st.subheader("Summary / Description")

        summary_text = "No summary available."

        # Try to extract from structured JSON
        if "summary" in result:
            summary_text = result["summary"].get("overall_gist", summary_text)
        elif "observations" in result and result["observations"]:
            summary_text = result["observations"][0].get("text", summary_text)
        else:
            trace = result.get("trace", [])
            for step in trace:
                actions = step.get("actions", [])
                for action in actions:
                    if action.get("step_id") == "synthesize_findings":
                        summary_text = action.get("result", {}).get("text", summary_text)
                        if summary_text:
                            break
                if summary_text != "No summary available.":
                    break

        st.write(summary_text)


