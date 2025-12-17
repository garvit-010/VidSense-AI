from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.tools.web_search import WebSearchTool
from app.llm.web_generator import WebAnswerGenerator
from app.ingestion.youtube import YouTubeTranscriptLoader
from app.chunking.splitter import DocumentChunker
from app.embeddings.embedder import EmbeddingProvider
from app.vectorstore.faiss_store import FAISSVectorStore
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.retriever import VectorRetriever
from app.llm.generator import AnswerGenerator
from app.memory.session import SessionMemory
from app.utils.logger import log_event

router = APIRouter()
web_search = WebSearchTool()
web_generator = WebAnswerGenerator()
memory = SessionMemory()


# ---------- Request Models ----------
class AskRequest(BaseModel):
    question: str
    session_id : str


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
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        vector_db = vector_store.load(embeddings)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="No video ingested yet")

    retriever = VectorRetriever(vector_db)

    # 🧠 MEMORY: get previous conversation
    memory_text = memory.get_context(payload.session_id)

    # 🧠 MEMORY-AWARE rewriting
    if memory_text:
        search_query = rewriter.rewrite_with_memory(
            payload.question,
            memory_text
        )
    else:
        search_query = rewriter.rewrite(payload.question)

    docs = retriever.retrieve(search_query)

    if not docs:
        answer = "Insufficient data in transcript"
        source = "transcript"
    else:
        context = "\n\n".join(doc.page_content for doc in docs)

        answer = generator.generate(
            context=context,
            question=payload.question
        )
        source = "transcript"

        if "Insufficient data in transcript" in answer:
            web_context = web_search.search_web(payload.question)
            web_context = web_context[:4000]

            answer = web_generator.generate(
                context=web_context,
                question=payload.question
            )
            source = "web"


    memory.add(payload.session_id, "user", payload.question)
    memory.add(payload.session_id, "assistant", answer)
    
    log_event({
        "session_id": payload.session_id,
        "question": payload.question,
        "answer": answer,
        "source": source,
        "used_memory": bool(memory_text),
        "num_retrieved_docs": len(docs) if docs else 0,
    })

    return {
        "question": payload.question,
        "answer": answer,
        "source": source
    }



