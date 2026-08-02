import os
import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    return os.path.splitext(filename)[0] + ".wav"



def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    audio = AudioSegment.from_file(input_path)

    audio = audio.set_channels(1).set_frame_rate(16000)

    audio.export(output_path, format="wav")

    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []
    for i, start in enumerate(range(0,len(audio),chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        base, _ = os.path.splitext(wav_path)
        chunk_path = f"{base}_chunk_{i}.wav"
        chunk.export(chunk_path,format='wav')

        chunks.append(chunk_path)

    return chunks


def cleanup_audio_files(files_to_clean: list):
    """Safely delete temporary audio files to prevent disk exhaustion in deployment."""
    if not files_to_clean:
        return
    for path in set(files_to_clean):
        try:
            if path and os.path.exists(path):
                os.remove(path)
                print(f"🧹 Cleaned up temporary file: {path}")
        except Exception as e:
            print(f"⚠️ Failed to remove temporary file {path}: {e}")


def process_input(source: str) -> tuple[list, list]:
    temp_files = []
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
        temp_files.append(wav_path)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)
        temp_files.append(wav_path)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    temp_files.extend(chunks)

    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks, temp_files


