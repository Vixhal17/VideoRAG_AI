from core.llm_factory import get_llm

#  Split text to small small chunks

def split_transcript(transcript: str) -> list:
  from langchain_text_splitters import RecursiveCharacterTextSplitter
  splitter = RecursiveCharacterTextSplitter(
    chunk_size = 3000,
    chunk_overlap = 200
  )

  return splitter.split_text(transcript)

def summarize(transcript: str) -> str:
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.runnables import RunnablePassthrough, RunnableLambda

  llm = get_llm()

  map_prompt = ChatPromptTemplate.from_messages(
    [
      ("system","Summarise this protion of a meeting transcript concisely. "),("human","{text}")
    ]
  )

  map_chain = map_prompt | llm | StrOutputParser()

  chunks = split_transcript(transcript)

  chunk_summaries = [map_chain.invoke({"text" : chunk}) for chunk in chunks]

  combined = "\n\n".join(chunk_summaries)

  combined_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI assistant. Combine the partial video summaries into one concise, well-structured summary. Preserve important technical details and use bullet points."
    ),
    ("human", "{text}")
  ])

  combined_chain = (
    RunnablePassthrough() | RunnableLambda(lambda x:{"text":x})| combined_prompt| llm | StrOutputParser()
  )

  return combined_chain.invoke(combined)

def generate_title(transcript: str) -> str:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda

    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an AI assistant that generates titles for videos.

            Generate a short, descriptive, and professional title based on the transcript.

            Rules:
            - Maximum 8 words.
            - Return ONLY the title.
            - Do not use quotation marks.
            - Do not add explanations.
            - Make it suitable as a YouTube or educational video title.
            """
        ),
        ("human", "{text}")
    ])

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | title_prompt
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])