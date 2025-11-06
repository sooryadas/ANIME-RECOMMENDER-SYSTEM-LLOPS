# src/recommender.py

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.prompt_template import get_anime_prompt


def _format_docs(docs):
    # Join retrieved docs into a single context string
    return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)


class AnimeRecommender:
    def __init__(self, retriever, api_key: str, model_name: str):
        # LLM
        self.llm = ChatGroq(api_key=api_key, model=model_name, temperature=0)

        # ChatPromptTemplate with {context} and {input}
        prompt: ChatPromptTemplate = get_anime_prompt()

        # RAG with runnables: retriever -> prompt -> LLM -> string
        self.chain = (
            {
                "context": retriever | _format_docs,
                "input": RunnablePassthrough(),   # key must match the prompt variable
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def get_recommendation(self, query: str) -> str:
        return self.chain.invoke(query)
