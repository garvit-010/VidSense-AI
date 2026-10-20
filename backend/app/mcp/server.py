"""
VidSense AI — Open-Source Model Context Protocol (MCP) Server.
Adheres strictly to the official Model Context Protocol (MCP) specification.
Exposes tools for video transcript semantic retrieval, deep web search,
and video metadata intelligence.
"""

from typing import Optional
from mcp.server.fastmcp import FastMCP
from app.embeddings.embedder import EmbeddingProvider
from app.vectorstore.chroma_store import ChromaVectorStore
from app.tools.web_search import WebSearchTool

# Initialize FastMCP Server
mcp_server = FastMCP(
    "vidsense-mcp-server",
    dependencies=["chromadb", "langchain", "duckduckgo-search"],
)

_embedder = None
_vector_store = None
_web_search = None


def get_services():
    global _embedder, _vector_store, _web_search
    if _embedder is None:
        _embedder = EmbeddingProvider()
    if _vector_store is None:
        _vector_store = ChromaVectorStore()
    if _web_search is None:
        _web_search = WebSearchTool()
    return _embedder, _vector_store, _web_search


@mcp_server.tool()
def search_transcript(query: str, video_id: Optional[str] = None, top_k: int = 4) -> str:
    """
    Search indexed YouTube transcripts using ChromaDB vector similarity.
    Args:
        query: Search keywords or natural language question.
        video_id: Optional YouTube video ID to filter chunks.
        top_k: Number of relevant transcript chunks to return (default: 4).
    """
    embedder, vector_store, _ = get_services()
    embeddings = embedder.get()

    try:
        results = vector_store.similarity_search(
            query=query, embeddings=embeddings, k=top_k, video_id=video_id
        )
        if not results:
            return "No matching transcript segments found in ChromaDB."

        formatted_chunks = []
        for i, doc in enumerate(results, start=1):
            v_id = doc.metadata.get("video_id", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", i)
            formatted_chunks.append(
                f"[Chunk {i} | Video: {v_id} | Index: {chunk_idx}]\n{doc.page_content.strip()}"
            )
        return "\n\n---\n\n".join(formatted_chunks)
    except Exception as e:
        return f"Error retrieving from ChromaDB: {str(e)}"


@mcp_server.tool()
def deep_web_research(query: str) -> str:
    """
    Perform deep web research to augment video knowledge with current facts,
    cross-references, and opinion perspectives using DuckDuckGo.
    Args:
        query: Search query for web intelligence.
    """
    _, _, web_search = get_services()
    try:
        result = web_search.search_web(query)
        return result[:3500] if result else "No relevant web results found."
    except Exception as e:
        return f"Error during web research: {str(e)}"


@mcp_server.tool()
def inspect_indexed_videos() -> str:
    """
    Inspect the ChromaDB collection to see how many chunks are indexed
    and system status.
    """
    embedder, vector_store, _ = get_services()
    try:
        embeddings = embedder.get()
        count = vector_store.count(embeddings)
        return f"VidSense ChromaDB Status: Active | Total Indexed Chunks: {count}"
    except Exception as e:
        return f"Error querying vector store status: {str(e)}"


if __name__ == "__main__":
    print("[VidSense MCP Server] Launching standard stdio MCP transport...")
    mcp_server.run()
