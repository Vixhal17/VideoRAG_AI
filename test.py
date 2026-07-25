from utils.audio_processor import process_input
from core.transcriber import transcribe_all

source = "http://www.youtube.com/watch?v=kD6Cigj6Fyc"

chunks = process_input(source)
print(transcribe_all(chunks))