from __future__ import annotations
import os

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "video_transcript"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings():
  from langchain_mistralai import MistralAIEmbeddings
  return MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
  )


def build_vector_store(transcript: str) -> Chroma:
  from chromadb import PersistentClient
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  from langchain_core.documents import Document
  from langchain_chroma import Chroma

  print("📚 Building Vector Store...")

  # Clear existing collection to avoid cross-video context pollution
  try:
      client = PersistentClient(path=CHROMA_DIR)
      client.delete_collection(COLLECTION_NAME)
      print("🧹 Cleared existing vector collection.")
  except Exception as e:
      print(f"⚠️ Note: Could not clear existing collection: {e}")

  splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
  )

  chunks = splitter.split_text(transcript)

  docs = [
    Document(page_content=chunk, metadata = {'chunk_index' : i})
    for i, chunk in enumerate(chunks)
  ]

  embeddings = get_embeddings()

  vector_store = Chroma.from_documents(
    documents = docs,
    embedding = embeddings,
    collection_name = COLLECTION_NAME,
    persist_directory = CHROMA_DIR
  )
  print("✅ Vector Store created successfully!")

  return vector_store


def load_vector_store() -> Chroma:
  from langchain_chroma import Chroma
  embeddings = get_embeddings()

  vector_store = Chroma(
    embedding_function = embeddings,
    collection_name = COLLECTION_NAME,
    persist_directory = CHROMA_DIR
  )
  return vector_store


def get_retriever(vector_store : Chroma, k : int = 4):
  return vector_store.as_retriever(
    search_type = 'similarity',
    search_kwargs = {'k':k}
  )