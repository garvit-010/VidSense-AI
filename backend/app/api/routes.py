from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.ingestion.youtube import YouTubeTranscriptLoader
from app.chunking.splitter import DocumentChunker
from app.embeddings.embedder import EmbeddingProvider
from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.retriever import VectorRetriever
from app.llm.generator import AnswerGenerator

router = APIRouter()


# ---------- Request Models ----------
class AskRequest(BaseModel):
    question: str


# ---------- Global Pipeline Setup ----------
embedder = EmbeddingProvider()
embeddings = embedder.get()

vector_store = FAISSVectorStore(index_path="faiss_main")
rewriter = QueryRewriter()
generator = AnswerGenerator()


# ---------- Routes ----------
@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/ingest/youtube/{video_id}")
def ingest_youtube(video_id: str):
    loader = YouTubeTranscriptLoader(video_id)
    docs = loader.load()

    chunker = DocumentChunker()
    chunks = chunker.split(docs)

    vector_store.build(chunks, embeddings)
    vector_store.save()

    return {
        "video_id": video_id,
        "chunks": len(chunks)
    }


@router.post("/ask")
def ask_question(payload: AskRequest):
    try:
        vector_db = vector_store.load(embeddings)
    except Exception:
        raise HTTPException(status_code=400, detail="No video ingested yet")

    retriever = VectorRetriever(vector_db)

    search_query = rewriter.rewrite(payload.question)
    docs = retriever.retrieve(search_query)

    context = "\n\n".join(doc.page_content for doc in docs)

    answer = generator.generate(
        context=context,
        question=payload.question
    )

    return {
        "question": payload.question,
        "answer": answer
    }
