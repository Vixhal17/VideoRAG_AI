from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize , generate_title
from core.extractor import extract_action_items, extract_key_decision,extract_questions
from core.rag_engine import build_rag_chain,ask_question
import sys
import traceback
load_dotenv()

def run_pipeline(source: str)->dict:
  print("Starting AI Video Assistant")

  chunks = process_input(source)

  transcript = transcribe_all(chunks)
  if not transcript.strip():
      raise ValueError("No speech was detected in the video.")
  
  print(f"\nRaw transcription (first 300 characters): \n{transcript[:300]}")

  title = generate_title(transcript)
  summary = summarize(transcript)

  action_items = extract_action_items(transcript)
  decisions = extract_key_decision(transcript)
  questions = extract_questions(transcript)

  print("\n📚 Building vector database...")
  rag_chain = build_rag_chain(transcript)
  print("✅ Vector database ready!")

  return {
      "title": title,
      "summary": summary,
      "transcript": transcript,
      "action_items": action_items,
      "decisions": decisions,
      "questions": questions,
      "rag_chain": rag_chain,
  }


if __name__ == "__main__":

    source = input("Enter YouTube URL or local file path: ").strip()

    try:
      result = run_pipeline(source)
    except Exception:
      traceback.print_exc()
      raise

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📝 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['decisions']}")
    print(f"\n❓Open Questions:\n{result['questions']}")
    print("=" * 60)


    # Phase 2 - Chat with your video via RAG
    print("\n💬 Chat with your video (type 'exit' to quit)\n")

    rag_chain = result["rag_chain"]

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        if not question:
            continue

        answer = ask_question(rag_chain, question)

        print(f"\n🤖 Assistant: {answer}\n")