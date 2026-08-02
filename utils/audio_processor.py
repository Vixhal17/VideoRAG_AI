import os
import yt_dlp
from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")

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
        "no_warnings": True,
        # Bypass YouTube cloud datacenter bot detection using mobile player clients
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "mweb", "web"],
                "player_skip": ["webpage", "configs", "js"],
            }
        },
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
            "Accept-Language": "en-US,en;q=0.9",
        },
    }

    # Optional: Support for cookies on cloud hosting
    cookies_path = os.getenv("YOUTUBE_COOKIES_PATH", "cookies.txt")
    if os.path.exists(cookies_path):
        ydl_opts["cookiefile"] = cookies_path
    elif os.getenv("YOUTUBE_COOKIES"):
        temp_cookie = os.path.join(DOWNLOAD_DIR, "yt_cookies.txt")
        with open(temp_cookie, "w", encoding="utf-8") as cf:
            cf.write(os.getenv("YOUTUBE_COOKIES"))
        ydl_opts["cookiefile"] = temp_cookie

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if "entries" in info:
            info = info["entries"][0]
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


