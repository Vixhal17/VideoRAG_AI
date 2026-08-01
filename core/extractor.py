#Actionable Items, decision, questions

from core.llm_factory import get_llm


def build_chain(system_prompt: str):
  from langchain_core.prompts import ChatPromptTemplate
  from langchain_core.output_parsers import StrOutputParser
  from langchain_core.runnables import RunnablePassthrough, RunnableLambda

  llm = get_llm(temperature=0.2)
  return (
    RunnablePassthrough() | RunnableLambda(lambda x: {"text" : x}) |
    ChatPromptTemplate.from_messages(
      [
        ("system",system_prompt),
        ("human","{text}")
      ]
    ) | llm | StrOutputParser()
  )


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        """
        You are an expert AI assistant.

        Analyze the video transcript and extract all actionable tasks or recommendations.

        For each action item, provide:
        - Task Description
        - Responsible Person (if mentioned, otherwise write "Not specified")
        - Deadline (if mentioned, otherwise write "Not specified")

        Format the output as a numbered list.

        If the transcript does not contain any action items, return:
        "No action items found."
        """
    )

    return chain.invoke(transcript)

def extract_key_decision(transcript: str) -> str:
    chain = build_chain(
        """
        You are an expert AI video analyst.

        Analyze the video transcript and extract the most important key takeaways, conclusions, or decisions presented by the speaker.

        Guidelines:
        - Focus only on the most important insights.
        - Remove repeated information.
        - Return the result as a numbered list.
        - Keep each point concise (1-2 sentences).

        If no key takeaways are found, return:
        "No key takeaways found."
        """
    )

    return chain.invoke(transcript)

def extract_questions(transcript: str) -> str:
    chain = build_chain(
        """
        You are an AI assistant.

        Analyze the video transcript and extract all questions mentioned by the speaker or any questions that naturally arise from the discussed topics.

        Guidelines:
        - Include explicit questions asked in the video.
        - Include important discussion questions if appropriate.
        - Return the result as a numbered list.

        If no questions are found, return:
        "No questions found."
        """
    )

    return chain.invoke(transcript)