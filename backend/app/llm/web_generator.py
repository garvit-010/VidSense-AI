import os
import re
from typing import List, Optional
from langchain_core.prompts import PromptTemplate
from app.config import HUGGINGFACEHUB_API_TOKEN, GOOGLE_API_KEY


class WebAnswerGenerator:
    """
    Web-augmented Answer Generator with circuit breaker and extractive synthesis fallback.
    """

    _gemini_circuit_open = True

    def __init__(self):
        self._fallback_llm = None
        self.prompt = PromptTemplate(
            template=(
                "Answer the question using ONLY the web content below.\n"
                "Be factual and concise.\n\n"
                "Web content:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question"],
        )

    @property
    def fallback_llm(self):
        if self._fallback_llm is None and GOOGLE_API_KEY and not WebAnswerGenerator._gemini_circuit_open:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self._fallback_llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.2,
                    max_retries=0,
                )
            except Exception:
                WebAnswerGenerator._gemini_circuit_open = True
        return self._fallback_llm

    def generate(self, context: str, question: str) -> str:
        prompt_text = self.prompt.format(context=context, question=question)

        # 1. Try Gemini
        if not WebAnswerGenerator._gemini_circuit_open and self.fallback_llm:
            try:
                response = self.fallback_llm.invoke(prompt_text)
                return response.content.strip()
            except Exception:
                WebAnswerGenerator._gemini_circuit_open = True

        # 2. Extract first 2 relevant lines from web context
        if not context:
            return "No web results found for this query."
        lines = [line.strip() for line in context.split("\n") if len(line.strip()) > 20]
        if lines:
            return " ".join(lines[:2])
        return f"Based on web research: {context[:250].strip()}..."
