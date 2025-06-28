import streamlit as st
import os
import tempfile
import base64
import requests
from gtts import gTTS
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
import av
import whisper
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Load Whisper model
whisper_model = whisper.load_model("base")

# Load FAISS vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Hugging Face API setup (Mistral)
HUGGINGFACE_API_KEY = st.secrets["hf_token"]
headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
    "Content-Type": "application/json"
}
api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

# Persistent session state
if "history" not in st.session_state:
    st.session_state.history = []

# WebRTC audio processor
class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame.to_ndarray().flatten())
        return frame

    def get_audio(self):
        if not self.frames:
            return None
        audio = np.concatenate(self.frames).astype("float32") / 32768.0
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            import soundfile as sf
            sf.write(f.name, audio, 48000)
            return f.name

# UI layout
st.set_page_config(page_title="Voice Assistant", layout="wide")
st.title("🎙️ Smart Voice Assistant with Mistral")
st.write("Ask anything using your voice!")

# WebRTC for live voice input
ctx = webrtc_streamer(
    key="speech",
    mode=WebRtcMode.SENDRECV,
    client_settings=ClientSettings(
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    ),
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

if ctx.audio_processor and st.button("Transcribe & Ask"):
    audio_path = ctx.audio_processor.get_audio()
    if audio_path:
        # Transcribe
        result = whisper_model.transcribe(audio_path)
        query = result["text"]
        st.session_state.history.append(("User", query))

        # Get context
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in docs])

        # Prepare Hugging Face request
        payload = {
            "inputs": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:",
            "parameters": {"max_new_tokens": 200}
        }
        response = requests.post(api_url, headers=headers, json=payload)
        result = response.json()

        try:
            answer = result[0]["generated_text"].split("Answer:")[-1].strip()
        except:
            answer = "Sorry, I couldn't understand."

        st.session_state.history.append(("AI", answer))

        # Text-to-speech
        tts = gTTS(answer)
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        st.audio(mp3_fp.getvalue(), format="audio/mp3")

# Show conversation
st.markdown("---")
st.subheader("🧠 Conversation History")
for role, text in st.session_state.history:
    st.markdown(f"**{role}:** {text}")
