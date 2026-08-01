import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None


def load_model():
    global _model

    if _model is None:
        import torch
        from faster_whisper import WhisperModel

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


def transcribe_chunk(chunk_path: str) -> str:
    model = load_model()

    segments, info = model.transcribe(
        chunk_path,
        task="translate",      # Always output English
        language=None,         # Auto-detect input language
        beam_size=5,           # Better accuracy
        vad_filter=True,
        temperature=0.0,
    )

    print(f"Detected language: {info.language}")

    text = "".join(segment.text for segment in segments)

    return text


def transcribe_all(chunks: list):
    full_transcript = ""

    for i, chunk in enumerate(chunks):
        print(f"🎤 Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk)
        full_transcript += text + " "

    print("✅ Transcription completed!")

    return full_transcript.strip()