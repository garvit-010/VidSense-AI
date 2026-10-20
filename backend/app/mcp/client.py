import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable


class MCPToolRegistry:
    """
    Model Context Protocol (MCP) Tool Registry and Client Manager.
    Allows VidSense agent to dynamically discover, inspect, and invoke
    tools provided by open-source MCP servers or local server adapters.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._default_config_path()
        self.registered_tools: Dict[str, Dict[str, Any]] = {}
        self._tool_handlers: Dict[str, Callable] = {}
        self._load_config()
        self._register_default_tools()

    def _default_config_path(self) -> str:
        base_dir = Path(__file__).resolve().parent.parent.parent
        return str(base_dir / "mcp_servers_config.json")

    def _load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"[MCPClient] Failed to load config from {self.config_path}: {e}")
                self.config = {}
        else:
            self.config = {
                "mcpServers": {
                    "vidsense-server": {
                        "command": "python",
                        "args": ["-m", "app.mcp.server"],
                        "description": "VidSense YouTube Video Intelligence & ChromaDB Search Server",
                    },
                    "web-research": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-duckduckgo"],
                        "description": "Open-source DuckDuckGo Web Intelligence MCP Server",
                    },
                    "filesystem-notes": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "./notes"],
                        "description": "Open-source Local Filesystem Context Provider",
                    },
                }
            }
            # Save default config if not exists
            try:
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(self.config, f, indent=2)
            except Exception:
                pass

    def _register_default_tools(self):
        """Registers VidSense built-in tools under the MCP contract."""
        from app.mcp.server import search_transcript, deep_web_research, inspect_indexed_videos

        self.register_tool(
            name="search_transcript",
            description="Search YouTube transcript segments in ChromaDB by semantic query.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text or question"},
                    "video_id": {"type": "string", "description": "Optional video ID filter"},
                    "top_k": {"type": "integer", "description": "Number of chunks (default: 4)"},
                },
                "required": ["query"],
            },
            handler=search_transcript,
        )

        self.register_tool(
            name="deep_web_research",
            description="Deep web search using DuckDuckGo to augment missing video details or latest news.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
            handler=deep_web_research,
        )

        self.register_tool(
            name="inspect_indexed_videos",
            description="Inspect the count of indexed video transcript chunks in ChromaDB.",
            parameters={"type": "object", "properties": {}},
            handler=inspect_indexed_videos,
        )

    def register_tool(self, name: str, description: str, parameters: Dict[str, Any], handler: Callable):
        self.registered_tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        self._tool_handlers[name] = handler

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns MCP-compliant tool specifications."""
        return list(self.registered_tools.values())

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Executes a tool call requested by the agent."""
        if tool_name not in self._tool_handlers:
            raise ValueError(f"Tool '{tool_name}' not found in MCP registry. Available: {list(self._tool_handlers.keys())}")
        
        handler = self._tool_handlers[tool_name]
        try:
            return handler(**arguments)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
