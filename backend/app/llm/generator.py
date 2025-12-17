from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from app.config import HUGGINGFACEHUB_API_TOKEN

class AnswerGenerator:
    def __init__(
        self,
        repo_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        temperature: float = 0.2,
    ):
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            max_new_tokens=512,
            temperature=temperature,
        )

        self.chat = ChatHuggingFace(llm=llm)

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
        response = self.chat.invoke(
            self.prompt.format(context=context, question=question)
        )
        return response.content.strip()
