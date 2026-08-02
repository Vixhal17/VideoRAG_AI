import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_local_model = None


def transcribe_chunk_groq(chunk_path: str, api_key: str) -> str:
    """Transcribe/translate audio chunk using Groq Whisper API (whisper-large-v3)."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        with open(chunk_path, "rb") as file:
            transcription = client.audio.translations.create(
                file=(os.path.basename(chunk_path), file.read()),
                model="whisper-large-v3",
            )
            return transcription.text
    except ImportError:
        # Fallback to direct HTTP request using requests
        import requests
        with open(chunk_path, "rb") as file:
            headers = {"Authorization": f"Bearer {api_key}"}
            files = {"file": (os.path.basename(chunk_path), file, "audio/wav")}
            data = {"model": "whisper-large-v3"}
            response = requests.post(
                "https://api.groq.com/openai/v1/audio/translations",
                headers=headers,
                files=files,
                data=data,
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("text", "")


def load_local_model():
    global _local_model

    if _local_model is None:
        import torch
        from faster_whisper import WhisperModel

        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            print(f"🚀 Loading local Faster-Whisper '{WHISPER_MODEL}' on GPU...")
        else:
            device = "cpu"
            compute_type = "int8"
            print(f"💻 Loading local Faster-Whisper '{WHISPER_MODEL}' on CPU...")

        _local_model = WhisperModel(
            WHISPER_MODEL,
            device=device,
            compute_type=compute_type,
        )

        print("✅ Local Whisper model loaded successfully!")

    return _local_model


def transcribe_chunk_local(chunk_path: str) -> str:
    model = load_local_model()

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


def transcribe_chunk(chunk_path: str) -> str:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key and groq_api_key.strip():
        try:
            print("⚡ Transcribing via Groq Whisper API (whisper-large-v3)...")
            return transcribe_chunk_groq(chunk_path, groq_api_key.strip())
        except Exception as e:
            print(f"⚠️ Groq Whisper API failed ({e}), falling back to local Whisper...")
            return transcribe_chunk_local(chunk_path)
    else:
        print("💻 Transcribing via local Faster-Whisper model...")
        return transcribe_chunk_local(chunk_path)


def transcribe_all(chunks: list) -> str:
    full_transcript = ""

    for i, chunk in enumerate(chunks):
        print(f"🎤 Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk)
        full_transcript += text + " "

    print("✅ Transcription completed!")

    return full_transcript.strip()