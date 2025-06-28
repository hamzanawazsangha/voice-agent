import streamlit as st
from streamlit_mic_recorder import mic_recorder
import whisper
from gtts import gTTS
from io import BytesIO
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Page settings
st.set_page_config(page_title="🎤 Voice Chat AI", layout="centered")
st.title("🎤 Voice Chat AI Assistant")
st.markdown("Talk to your assistant using your microphone. Ask any question from your data!")

# Load Whisper model once
@st.cache_resource
@st.cache_resource
def load_or_build_vectorstore():
    if os.path.exists("faiss_index/index.faiss"):
        return FAISS.load_local("faiss_index", embedding).as_retriever()
    
    loader = TextLoader("arslan_faqs.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, embedding)
    vectorstore.save_local("faiss_index")

    return vectorstore.as_retriever()
    
def load_whisper_model():
    return whisper.load_model("base")

# Load FAISS retriever
@st.cache_resource
def load_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs = FAISS.load_local("faiss_index", embedding, allow_dangerous_deserialization=True)
    return vs.as_retriever()

# Load QA Chain using HuggingFace LLM
@st.cache_resource
def load_qa_chain():
    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", model_kwargs={"temperature": 0.5, "max_new_tokens": 512})
    retriever = load_vectorstore()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

model = load_whisper_model()
qa_chain = load_qa_chain()

# Record voice
st.subheader("🎙️ Record Your Question")
audio = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="⏹️ Stop Recording", just_once=True, key="recorder")

if audio:
    # Save audio file
    with open("query.wav", "wb") as f:
        f.write(audio["bytes"])
    st.audio("query.wav", format="audio/wav")

    # Transcribe
    st.info("🔎 Transcribing...")
    result = model.transcribe("query.wav")
    query = result["text"]
    st.success(f"🗣️ You said: {query}")

    # Answer
    st.info("💬 Generating response...")
    response = qa_chain.run(query)
    st.success("✅ Response generated!")

    st.subheader("🤖 Answer:")
    st.markdown(f"**{response}**")

    # Speak
    tts = gTTS(response)
    tts_audio = BytesIO()
    tts.write_to_fp(tts_audio)
    st.audio(tts_audio.getvalue(), format="audio/mp3")
