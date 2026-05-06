import http.client
import json
import os

import deepl
import streamlit as st
import whisper
from dotenv import load_dotenv

from alpha.agent import MusicAgentChat

# --- Configuration & Initialization ---
load_dotenv()

st.set_page_config(page_title="AI Music Generator", page_icon="🎵")


# Secrets handling (Streamlit Cloud uses st.secrets, local uses .env)
def get_secret(key, default=None):
    return st.secrets.get(key, os.getenv(key, default))


GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
DEEPL_API_KEY = get_secret("DEEPL_API_KEY")
MUSICAPI_API_KEY = get_secret("MUSICAPI_API_KEY")

if not all([GEMINI_API_KEY, DEEPL_API_KEY, MUSICAPI_API_KEY]):
    st.error("Missing API keys! Please check your .env file or Streamlit Secrets.")
    st.stop()


# Initialize models
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("large-v3-turbo")
    translator = deepl.Translator(DEEPL_API_KEY)
    agent = MusicAgentChat()
    return whisper_model, translator, agent


whisper_model, translator, agent = load_models()

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {"history": []}
if "last_audio_url" not in st.session_state:
    st.session_state.last_audio_url = None


# --- Helper Functions ---
def generate_music(file_bytes, filename, title, language, genre):
    """Ported logic from alpha/main.py"""
    try:
        # Save temporary file for Whisper
        temp_path = f"temp_{filename}"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        with st.status("Processing audio...", expanded=True) as status:
            st.write("Transcribing with Whisper...")
            extracted_lyrics = whisper_model.transcribe(temp_path)
            original_lyrics = extracted_lyrics["text"]

            st.write("Translating lyrics...")
            translated_lyrics = translator.translate_text(
                original_lyrics, target_lang=language
            )

            st.write("Calling MusicAPI...")
            conn = http.client.HTTPSConnection("api.musicapi.ai")
            payload = json.dumps(
                {
                    "custom_mode": True,
                    "prompt": str(translated_lyrics),
                    "title": title,
                    "tags": genre,
                    "negative_tags": "piano",
                    "gpt_description_prompt": "",
                    "make_instrumental": False,
                    "mv": "sonic-v3-5",
                }
            )
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {MUSICAPI_API_KEY}",
            }
            conn.request("POST", "/api/v1/sonic/create", payload, headers)
            res = conn.getresponse()
            data = json.loads(res.read())

            if data.get("code") == 200 and "data" in data:
                task_id = data["data"].get("task_id")
            else:
                return {"error": f"API Error: {data}"}

            # Simple polling (in a real app, you might want more robust polling)
            st.write("Generating music (this may take a minute)...")
            import time

            for _ in range(10):  # 10 attempts
                time.sleep(10)
                conn.request("GET", f"/api/v1/sonic/task/{task_id}", "", headers)
                res = conn.getresponse()
                task_data = json.loads(res.read())
                if task_data.get("code") == 200 and "data" in task_data:
                    audio_url = next(
                        (
                            item["audio_url"]
                            for item in task_data["data"]
                            if "audio_url" in item
                        ),
                        None,
                    )
                    if audio_url:
                        status.update(label="Music Generated!", state="complete")
                        return {"audio_url": audio_url, "success": True}

            return {"error": "Timeout waiting for music generation"}

    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# --- UI Layout ---
st.title("🎵 AI Music Generator Chat")
st.markdown("Talk to the agent to set up your song, upload a file, and generate music!")

# Sidebar for file upload and parameters
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader(
        "Upload an audio file (lyrics source)", type=["mp3", "wav", "m4a"]
    )

    if uploaded_file:
        st.info(f"File uploaded: {uploaded_file.name}")
        st.session_state.agent_state["file_uploaded"] = True
        st.session_state.agent_state["filename"] = uploaded_file.name

    st.divider()
    st.subheader("Current Parameters")

    def get_val(p):
        v = st.session_state.agent_state.get(p)
        if isinstance(v, dict):
            return v.get("value", "Not set")
        return v or "Not set"

    st.write(f"**Title:** {get_val('title')}")
    st.write(f"**Genre:** {get_val('genre')}")
    st.write(f"**Language:** {get_val('language')}")

    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.session_state.agent_state = {"history": []}
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's the song title? / Set the genre to rock..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with agent
    with st.chat_message("assistant"):
        # Update internal agent state
        st.session_state.agent_state = agent.update_state(
            st.session_state.agent_state, prompt
        )

        # Get agent response
        response_data = agent.chat(
            prompt, st.session_state.agent_state.get("history", [])
        )
        reply = response_data.get("reply", "I'm not sure how to respond.")
        action = response_data.get("action", "chat")

        # Check for missing params (similar to Flask logic)
        def has_param(p):
            v = st.session_state.agent_state.get(p)
            return bool(v.get("value")) if isinstance(v, dict) else bool(v)

        missing = [p for p in ("title", "genre", "language") if not has_param(p)]

        if action == "generate_music" or (not missing and uploaded_file):
            if not uploaded_file:
                reply = "I have all the details, but I still need you to upload an audio file in the sidebar! 📁"
            else:
                st.markdown(reply)
                # Auto-trigger generation if all info is present
                title = st.session_state.agent_state.get("title", {}).get(
                    "value", "New Song"
                )
                genre = st.session_state.agent_state.get("genre", {}).get(
                    "value", "pop"
                )
                lang = st.session_state.agent_state.get("language", {}).get(
                    "value", "EN"
                )

                result = generate_music(
                    uploaded_file.getvalue(), uploaded_file.name, title, lang, genre
                )

                if "audio_url" in result:
                    reply = f"✅ Success! Your song '{title}' is ready!"
                    st.session_state.last_audio_url = result["audio_url"]
                else:
                    reply = f"❌ Error: {result.get('error')}"

        elif missing and action != "request_param":
            ask_for = missing[0]
            friendly = {"title": "song title", "genre": "genre", "language": "language"}
            reply = f"I've got some details, but I still need the **{friendly[ask_for]}**. What should it be?"

        st.markdown(reply)

        if st.session_state.last_audio_url:
            st.audio(st.session_state.last_audio_url)
            st.session_state.last_audio_url = None  # Reset after showing

        # Update history
        st.session_state.agent_state["history"].append(
            {"role": "user", "content": prompt}
        )
        st.session_state.agent_state["history"].append(
            {"role": "assistant", "content": reply}
        )
        st.session_state.messages.append({"role": "assistant", "content": reply})

# Handle the "Ready to generate" state if file was uploaded but chat hasn't moved
if uploaded_file and not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "I see you've uploaded a file! Let's set up the title, genre, and language for your remix. What should the song title be?"
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "I see you've uploaded a file! Let's set up the title, genre, and language for your remix. What should the song title be?",
            }
        )
