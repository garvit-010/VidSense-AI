import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import re

load_dotenv()


class QueryRewriter:
    def __init__(
        self,
        repo_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        temperature: float = 0.0,
        max_new_tokens: int = 64,
    ):
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        self.chat = ChatHuggingFace(llm=self.llm)

        self.prompt = PromptTemplate(
            template=(
                "You are a search query optimizer.\n"
                "Rewrite the user question into a short, keyword-focused search query.\n"
                "Rules:\n"
                "- Do NOT answer the question\n"
                "- Do NOT explain\n"
                "- Do NOT use full sentences\n"
                "- Do NOT use markdown\n"
                "- Output keywords only\n\n"
                "User question: {question}\n"
                "Search query:"
            ),
            input_variables=["question"],
        )

    def rewrite(self, question: str) -> str:
        try:
            response = self.chat.invoke(
                self.prompt.format(question=question)
            )

            import re
            text = response.content.strip()
            text = text.replace("Search query:", "")
            text = re.sub(r"[.,]", "", text)
            text = re.sub(r"\s+", " ", text).strip()

            # keep only first N meaningful words
            words = text.split(" ")
            return " ".join(words[:10])
        except Exception:
            return question
    def rewrite_with_memory(self, question: str, memory: str) -> str:
        combined = f"""
                    Conversation context:
                    {memory}

                    Current question:
                    {question}
                    """
        return self.rewrite(combined)



