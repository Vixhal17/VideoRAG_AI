import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough , RunnableLambda
from core.vector_Store import build_vector_store,load_vector_store,get_retriever

def get_llm():
  return ChatMistralAI(
    model = 'mistral-small-latest',
    mistral_api_key = os.getenv("MISTRAL_API_KEY"),
    temperature=0.3
  )

def format_docs(docs):
  return "\n\n\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(transcript : str):

  vector_store = build_vector_store(transcript)

  retriever = get_retriever(vector_store , k = 4)

  llm = get_llm()

  prompt = ChatPromptTemplate.from_messages(
    [
        ("system","""
              You are an AI Video Assistant.
              Answer the user's question using ONLY the provided transcript context.
              If the answer cannot be found in the transcript, reply:
              "I couldn't find that information in the video."
              Be concise and accurate.
              Transcript Context : {context}
              """
        ),
        ("human", "{question}")
    ]
  )

  # full LCEL Rag Pipeline
  print(type(retriever))
  print(type(prompt))
  print(type(llm))
  a = {
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough(),
    }

  print("Step 1 OK")

  b = a | prompt
  print("Step 2 OK")

  c = b | llm
  print("Step 3 OK")

  rag_chain = c | StrOutputParser()
  print("Step 4 OK")

  return rag_chain



def load_rag_chain():

    vector_store = load_vector_store()

    retriever = get_retriever(vector_store, k=4)

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
                """
                  You are an AI Video Assistant.
                  Answer the user's question using ONLY the provided transcript context.
                  If the answer is not present in the transcript, reply:
                  "I couldn't find that information in the video."
                  Context : {context}
                  """
            ),
            ("human", "{question}")
        ]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain



def ask_question(rag_chain, question: str) -> str:
    try:
        answer = rag_chain.invoke(question)

        return answer

    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while processing your question."