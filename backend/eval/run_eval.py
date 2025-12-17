import json
import pandas as pd
from datasets import Dataset
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import os

from ragas.metrics import (
    ContextRelevance,
    Faithfulness,
    AnswerRelevancy,
)
from ragas import evaluate

from app.api.routes import (
    rewriter,
    vector_store,
    embeddings,
    generator,
    web_search,
    web_generator,
)
from app.retrieval.retriever import VectorRetriever

# Increased max_new_tokens to avoid LLMDidNotFinishException
evaluator_llm = LangchainLLMWrapper(
    ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            task="text-generation",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            max_new_tokens=1024, 
            temperature=0,
        )
    )
)

evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

def ask_system(question: str):
    vector_db = vector_store.load(embeddings)
    retriever = VectorRetriever(vector_db)

    search_query = rewriter.rewrite(question)
    docs = retriever.retrieve(search_query)
    context = "\n\n".join(d.page_content for d in docs)

    answer = generator.generate(context=context, question=question)
    source = "transcript"

    if "Insufficient data in transcript" in answer:
        web_ctx = web_search.search_web(question)[:4000]
        answer = web_generator.generate(context=web_ctx, question=question)
        context = web_ctx
        source = "web"

    return answer, context, source

def main():
    with open("eval/sample_eval.json") as f:
        data = json.load(f)

    rows = []
    for item in data:
        answer, context, source = ask_system(item["question"])

        rows.append({
            "user_input": item["question"],
            "response": answer,
            "reference": item["ground_truth"],
            "retrieved_contexts": [context],
        })

    dataset = Dataset.from_pandas(pd.DataFrame(rows))

    results = evaluate(
        dataset,
        metrics=[
            ContextRelevance(),
            Faithfulness(),
            AnswerRelevancy(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # Convert to DataFrame for a cleaner output in your terminal
    df = results.to_pandas()
    print("\n--- Evaluation Results ---")
    print(df[['user_input', 'context_relevance', 'faithfulness', 'answer_relevancy']])
    print("\n--- Summary Scores ---")
    print(results)

if __name__ == "__main__":
    main()