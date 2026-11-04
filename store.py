#Enterprise RAG - Embeddings and Vector Store
from config import EMBED_MODEL, CHROMA_DIR, COLLECTION_NAME
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 🧠Embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# 📦Vector store
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)
