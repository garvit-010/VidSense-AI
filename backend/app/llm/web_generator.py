from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import os

class WebAnswerGenerator:
    def __init__(self):
        self.llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                task="text-generation",
                huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
                max_new_tokens=256,
                temperature=0.2,
            )
        )

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

    def generate(self, context: str, question: str) -> str:
        response = self.llm.invoke(
            self.prompt.format(context=context, question=question)
        )
        return response.content.strip()
