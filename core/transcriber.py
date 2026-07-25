import os
import torch
from faster_whisper import WhisperModel

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None


def load_model():
    global _model

    if _model is None:

        # Automatically choose GPU if available
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            print(f"🚀 Loading Faster-Whisper '{WHISPER_MODEL}' on GPU...")
        else:
            device = "cpu"
            compute_type = "int8"
            print(f"💻 Loading Faster-Whisper '{WHISPER_MODEL}' on CPU...")

        _model = WhisperModel(
            WHISPER_MODEL,
            device=device,
            compute_type=compute_type,
        )

        print("✅ Model loaded successfully!")

    return _model


def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    model = load_model()

    task = "translate" if translate else "transcribe"

    segments, info = model.transcribe(
        chunk_path,
        task=task,
        language=None,      # Auto detect language
        beam_size=1,        # Faster decoding
        vad_filter=True     # Skip silent regions
    )

    text = "".join(segment.text for segment in segments)

    return text


def transcribe_all(chunks: list, translate: bool = False):
    full_transcript = ""

    for i, chunk in enumerate(chunks):
        print(f"🎤 Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, translate)

        full_transcript += text + " "

    print("✅ Transcription completed!")

    return full_transcript.strip()