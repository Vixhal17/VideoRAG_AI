import os
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "video_transcript"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def get_embeddings():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  return HuggingFaceEmbeddings(
    model_name = EMBEDDING_MODEL,
    model_kwargs = {"device" : device},
    encode_kwargs={"normalize_embeddings": True}
  )


def build_vector_store(transcript: str) -> Chroma:
  print("📚 Building Vector Store...")

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


def load_vector_store()->Chroma:
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