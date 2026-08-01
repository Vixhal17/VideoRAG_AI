from core.llm_factory import get_llm

def format_docs(docs):
  return "\n\n\n\n".join([doc.page_content for doc in docs])


def _create_rag_chain(retriever):
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.runnables import RunnablePassthrough, RunnableLambda

  llm = get_llm(temperature=0.3)

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


def build_rag_chain(transcript : str):
  from core.vector_Store import build_vector_store, get_retriever
  vector_store = build_vector_store(transcript)
  retriever = get_retriever(vector_store , k = 4)
  return _create_rag_chain(retriever)


def load_rag_chain():
    from core.vector_Store import load_vector_store, get_retriever
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=4)
    return _create_rag_chain(retriever)


def ask_question(rag_chain, question: str) -> str:
    try:
        answer = rag_chain.invoke(question)

        return answer

    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while processing your question."