from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ingestion.youtube import YouTubeTranscriptLoader
from app.chunking.splitter import DocumentChunker
from app.embeddings.embedder import EmbeddingProvider
from app.vectorstore.chroma_store import ChromaVectorStore
from app.agent.orchestrator import VideoIntelligenceAgent, AgentResponse
from app.memory.session import SessionMemory
from app.tools.web_search import WebSearchTool
from app.mcp.client import MCPToolRegistry

router = APIRouter()

# Global Singleton Services
embedder = EmbeddingProvider()
embeddings = embedder.get()
chroma_store = ChromaVectorStore(persist_directory="chroma_db")
session_memory = SessionMemory()
mcp_registry = MCPToolRegistry()

# Autonomous Agent
agent = VideoIntelligenceAgent(
    vector_store=chroma_store,
    embeddings=embeddings,
    memory=session_memory,
)


class AskRequest(BaseModel):
    question: str
    session_id: str
    video_id: Optional[str] = None


class MCPCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


@router.get("/health")
def health():
    try:
        indexed_chunks = chroma_store.count(embeddings)
    except Exception:
        indexed_chunks = 0

    return {
        "status": "healthy",
        "system": "VidSense AI Autonomous Agent",
        "vector_store": "ChromaDB",
        "indexed_chunks": indexed_chunks,
        "mcp_tools_available": [t["name"] for t in mcp_registry.list_tools()],
    }


@router.post("/ingest/youtube/{video_id}")
def ingest_youtube(video_id: str):
    """
    Ingest and index a YouTube video transcript into ChromaDB with deterministic chunk metadata.
    """
    try:
        loader = YouTubeTranscriptLoader(video_id)
        docs = loader.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch transcript: {str(e)}")

    chunker = DocumentChunker()
    chunks = chunker.split(docs)

    # Ingest into ChromaDB with video_id metadata
    chroma_store.build(documents=chunks, embeddings=embeddings, video_id=video_id)

    return {
        "status": "success",
        "video_id": video_id,
        "chunks_indexed": len(chunks),
        "vector_store": "ChromaDB",
    }


@router.post("/ask", response_model=AgentResponse)
def ask_question(payload: AskRequest):
    """
    Execute full agentic workflow: Guardrails -> Query Rewriting -> ChromaDB Semantic Retrieval
    -> Sufficiency Check -> Autonomous MCP Tool Execution -> Output Validation -> MLflow Telemetry.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    response = agent.run(
        question=payload.question,
        session_id=payload.session_id,
        video_id=payload.video_id,
    )
    return response


@router.get("/mcp/tools")
def list_mcp_tools():
    """List all tools registered in the Model Context Protocol (MCP) registry."""
    return {"tools": mcp_registry.list_tools()}


@router.post("/mcp/call")
def call_mcp_tool(payload: MCPCallRequest):
    """Directly execute a Model Context Protocol (MCP) tool."""
    try:
        result = mcp_registry.call_tool(payload.tool_name, payload.arguments)
        return {"tool": payload.tool_name, "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
