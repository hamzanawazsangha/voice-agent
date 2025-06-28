import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import whisper
import numpy as np
from gtts import gTTS
from io import BytesIO
import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# Load Whisper model
model = whisper.load_model("base")

# Load vector store
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Hugging Face API key from secrets
HUGGINGFACEHUB_API_TOKEN = st.secrets["hf_token"]

# Load Mistral model via Hugging Face
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512},
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# App UI
st.set_page_config(page_title="Voice AI Assistant", page_icon="🎙️", layout="centered")
st.title("🎙️ AI Voice Assistant with Knowledge Retrieval")
st.markdown("Speak into the mic and get intelligent responses from your uploaded knowledge base!")

# Audio buffer
audio_buffer = []

# Audio processing callback
def audio_callback(frame):
    audio = frame.to_ndarray()
    audio_buffer.append(audio)
    return av.AudioFrame.from_ndarray(audio, layout="mono")

# WebRTC streamer
ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    in_audio_callback=audio_callback,
    media_stream_constraints={"audio": True, "video": False},
)

# Button to trigger processing
if st.button("🔍 Transcribe and Answer"):
    if not audio_buffer:
        st.warning("No audio captured. Please speak first.")
    else:
        audio_data = np.concatenate(audio_buffer, axis=0).astype(np.float32)
        whisper_audio = whisper.pad_or_trim(audio_data.flatten())
        mel = whisper.log_mel_spectrogram(whisper_audio).to(model.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
        query = result.text
        st.write(f"**You said:** {query}")

        with st.spinner("Thinking..."):
            response = qa_chain.run(query)
        st.success("Response generated!")

        st.markdown(f"**🤖 AI Response:** {response}")

        # Text-to-speech
        tts = gTTS(text=response, lang='en')
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        st.audio(mp3_fp.getvalue(), format="audio/mp3")

        # Reset buffer
        audio_buffer.clear()
