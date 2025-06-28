from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

@st.cache_resource
def build_vectorstore():
    loader = TextLoader("your_docs.txt")  # or PDFLoader, etc.
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(splits, embedding)
    
    # Save for future use (optional)
    vectorstore.save_local("faiss_index")
    
    return vectorstore.as_retriever()
