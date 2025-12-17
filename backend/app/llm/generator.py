from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from app.config import GOOGLE_API_KEY


class AnswerGenerator:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.2,
    ):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
        )

        self.prompt = PromptTemplate(
            template=(
                "You are a factual AI assistant answering questions based ONLY on the provided context.\n"
                "Rules:\n"
                "- Use ONLY the context below\n"
                "- If the answer is not present, say: \"Insufficient data in transcript\"\n"
                "- Do NOT use outside knowledge\n"
                "- Be concise and factual\n\n"
                "Context:\n"
                "{context}\n\n"
                "Question:\n"
                "{question}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question"],
        )

    def generate(self, context: str, question: str) -> str:
        response = self.llm.invoke(
            self.prompt.format(context=context, question=question)
        )
        return response.content.strip()
