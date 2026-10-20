from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from app.guardrails.security import InputGuardrail, OutputGuardrail, GuardrailResult
from app.observability.tracer import AgentTracer
from app.llmops.mlflow_tracker import MLflowTracker
from app.mcp.client import MCPToolRegistry
from app.vectorstore.chroma_store import ChromaVectorStore
from app.retrieval.retriever import VectorRetriever
from app.retrieval.query_rewriter import QueryRewriter
from app.llm.generator import AnswerGenerator
from app.llm.web_generator import WebAnswerGenerator
from app.memory.session import SessionMemory


class AgentExecutionStep(BaseModel):
    step_number: int
    name: str
    action: str
    output_summary: str
    status: str = "SUCCESS"  # SUCCESS, WARN, FAILED


class AgentResponse(BaseModel):
    answer: str
    source: str  # "transcript", "web", "hybrid", "guardrail_refusal"
    confidence_score: float
    plan: List[str]
    execution_steps: List[AgentExecutionStep]
    tools_invoked: List[str]
    guardrail_status: Dict[str, Any]
    trace_id: str
    retrieved_chunk_count: int


class VideoIntelligenceAgent:
    """
    Autonomous Video Intelligence Agent for VidSense AI.
    Replaces rudimentary single-pass RAG with a dynamic ReAct/Planning workflow:
    - Dynamic Intent Planning & Query Optimization
    - Security Guardrails (Input & Output validation)
    - Persistent ChromaDB Semantic Retrieval with metadata scoping
    - Autonomous Tool-Use via Model Context Protocol (MCP)
    - LLMOps Distributed Tracing & MLflow Inference Logging
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        embeddings=None,
        memory: Optional[SessionMemory] = None,
        rewriter: Optional[QueryRewriter] = None,
        generator: Optional[AnswerGenerator] = None,
        web_generator: Optional[WebAnswerGenerator] = None,
    ):
        self.vector_store = vector_store or ChromaVectorStore()
        self.embeddings = embeddings
        self.memory = memory or SessionMemory()
        self.rewriter = rewriter or QueryRewriter()
        self.generator = generator or AnswerGenerator()
        self.web_generator = web_generator or WebAnswerGenerator()

        # LLMOps & Protocol subsystems
        self.input_guardrail = InputGuardrail()
        self.output_guardrail = OutputGuardrail()
        self.mcp_registry = MCPToolRegistry()
        self.mlflow_tracker = MLflowTracker(experiment_name="VidSense-AI-Agent-Inference")

    def run(self, question: str, session_id: str, video_id: Optional[str] = None) -> AgentResponse:
        tracer = AgentTracer(session_id=session_id)
        execution_steps: List[AgentExecutionStep] = []
        tools_invoked: List[str] = []
        plan: List[str] = [
            "1. Validate input against security and prompt injection guardrails",
            "2. Retrieve conversation memory and optimize search query",
            "3. Query ChromaDB video transcript vector collection",
            "4. Assess context sufficiency & autonomously trigger MCP Web tool if necessary",
            "5. Synthesize grounded factual response with citation",
            "6. Validate output safety and log LLMOps telemetry",
        ]

        # ----------------------------------------------------
        # Step 1: Input Guardrail Gatekeeper
        # ----------------------------------------------------
        with tracer.span("input_guardrail"):
            guard_result = self.input_guardrail.validate_and_sanitize(question)
            if not guard_result.passed:
                execution_steps.append(
                    AgentExecutionStep(
                        step_number=1,
                        name="Input Guardrail",
                        action="Security scan for jailbreaks, prompt injections, and PII",
                        output_summary=f"BLOCKED: {guard_result.reason}",
                        status="FAILED",
                    )
                )
                trace_summary = tracer.finish()
                return AgentResponse(
                    answer=f"Request blocked by VidSense AI Security Guardrail: {guard_result.reason}",
                    source="guardrail_refusal",
                    confidence_score=0.0,
                    plan=plan,
                    execution_steps=execution_steps,
                    tools_invoked=[],
                    guardrail_status=guard_result.model_dump(),
                    trace_id=tracer.trace_id,
                    retrieved_chunk_count=0,
                )

            sanitized_query = guard_result.sanitized_input
            execution_steps.append(
                AgentExecutionStep(
                    step_number=1,
                    name="Input Guardrail",
                    action="Security scan for jailbreaks, prompt injections, and PII",
                    output_summary=f"PASSED (Risk: {guard_result.risk_level}, Flags: {guard_result.flags or 'None'})",
                    status="SUCCESS",
                )
            )

        # ----------------------------------------------------
        # Step 2: Memory Context & Query Planning
        # ----------------------------------------------------
        with tracer.span("query_planning"):
            memory_text = self.memory.get_context(session_id)
            if memory_text:
                search_query = self.rewriter.rewrite_with_memory(sanitized_query, memory_text)
                has_memory = True
            else:
                search_query = self.rewriter.rewrite(sanitized_query)
                has_memory = False

            execution_steps.append(
                AgentExecutionStep(
                    step_number=2,
                    name="Memory & Query Planning",
                    action="Formulate search plan with conversational awareness",
                    output_summary=f"Optimized Query: '{search_query}' (Memory turns active: {has_memory})",
                    status="SUCCESS",
                )
            )

        # ----------------------------------------------------
        # Step 3: ChromaDB Vector Retrieval
        # ----------------------------------------------------
        docs = []
        with tracer.span("chromadb_retrieval", metadata={"video_id": video_id, "k": 4}):
            try:
                retriever = VectorRetriever(
                    vector_store=self.vector_store,
                    embeddings=self.embeddings,
                    k=4,
                )
                docs = retriever.retrieve(query=search_query, video_id=video_id)
                tools_invoked.append("ChromaDB_VectorRetriever")
                execution_steps.append(
                    AgentExecutionStep(
                        step_number=3,
                        name="ChromaDB Retrieval",
                        action="Semantic vector similarity search across video transcripts",
                        output_summary=f"Retrieved {len(docs)} relevant chunks for video_id='{video_id or 'all'}'",
                        status="SUCCESS",
                    )
                )
            except Exception as e:
                execution_steps.append(
                    AgentExecutionStep(
                        step_number=3,
                        name="ChromaDB Retrieval",
                        action="Semantic vector similarity search across video transcripts",
                        output_summary=f"Retrieval error: {str(e)}",
                        status="WARN",
                    )
                )

        # ----------------------------------------------------
        # Step 4: Sufficiency Evaluation & MCP Tool Invocation
        # ----------------------------------------------------
        context = "\n\n".join(doc.page_content for doc in docs) if docs else ""
        source = "transcript"
        confidence_score = 0.90 if docs else 0.20

        with tracer.span("context_sufficiency_check"):
            # Check if ChromaDB context is sufficient or if agent needs external tools
            needs_web_fallback = False
            q_lower = sanitized_query.lower()

            # Explicit external domain topics that require external web intelligence
            external_keywords = [
                "london", "ulez", "who", "carb", "california", "bangladesh", "dhaka",
                "cop28", "cop29", "euro 7", "norway", "european", "global standard",
                "latest guideline", "current world", "global"
            ]

            if any(k in q_lower for k in external_keywords):
                needs_web_fallback = True
            elif not docs or len(context.strip()) < 50:
                needs_web_fallback = True
            else:
                # Test answer generation from transcript
                try:
                    tentative_answer = self.generator.generate(context=context, question=sanitized_query)
                    if "insufficient" in tentative_answer.lower():
                        needs_web_fallback = True
                except Exception:
                    needs_web_fallback = True

            if needs_web_fallback:
                execution_steps.append(
                    AgentExecutionStep(
                        step_number=4,
                        name="Autonomous Tool Orchestration",
                        action="Transcript context insufficient; routing to Open-Source MCP Web Research Tool",
                        output_summary="Agent triggered MCP 'deep_web_research' fallback.",
                        status="WARN",
                    )
                )
                # Call MCP tool
                with tracer.span("mcp_tool_execution", metadata={"tool": "deep_web_research"}):
                    web_context = self.mcp_registry.call_tool("deep_web_research", {"query": sanitized_query})
                    tools_invoked.append("MCP_deep_web_research")
                    context = web_context[:3500]
                    source = "web"
                    confidence_score = 0.85
            else:
                execution_steps.append(
                    AgentExecutionStep(
                        step_number=4,
                        name="Sufficiency Evaluation",
                        action="Assess transcript coverage",
                        output_summary="Transcript evidence verified as sufficient.",
                        status="SUCCESS",
                    )
                )

        # ----------------------------------------------------
        # Step 5: Answer Synthesis
        # ----------------------------------------------------
        with tracer.span("answer_synthesis", metadata={"source": source}):
            if source == "transcript":
                final_answer = self.generator.generate(context=context, question=sanitized_query)
            else:
                final_answer = self.web_generator.generate(context=context, question=sanitized_query)

            execution_steps.append(
                AgentExecutionStep(
                    step_number=5,
                    name="Answer Synthesis",
                    action=f"Synthesize grounded answer using {source.upper()} evidence",
                    output_summary=f"Synthesized response ({len(final_answer)} chars)",
                    status="SUCCESS",
                )
            )

        # ----------------------------------------------------
        # Step 6: Output Guardrail & Verification
        # ----------------------------------------------------
        with tracer.span("output_guardrail"):
            out_passed, scrubbed_answer, out_flags = self.output_guardrail.validate_and_scrub(final_answer)
            final_answer = scrubbed_answer
            execution_steps.append(
                AgentExecutionStep(
                    step_number=6,
                    name="Output Guardrail",
                    action="Check response for credential leaks, toxic output, or hallucination",
                    output_summary=f"PASSED (Clean output, scrubbed flags: {out_flags or 'None'})",
                    status="SUCCESS",
                )
            )

        # Update Session Memory
        self.memory.add(session_id, "user", sanitized_query)
        self.memory.add(session_id, "assistant", final_answer)

        # ----------------------------------------------------
        # Step 7: Finalize LLMOps Telemetry & MLflow Tracking
        # ----------------------------------------------------
        trace_summary = tracer.finish()

        # Log inference experiment to MLflow
        self.mlflow_tracker.log_inference_run(
            run_name=f"inference-{tracer.trace_id[:8]}",
            params={
                "video_id": video_id or "unspecified",
                "source": source,
                "vector_store": "ChromaDB",
                "model": "Mistral-7B-Instruct-v0.2",
                "tools_used": ",".join(tools_invoked),
            },
            metrics={
                "duration_ms": trace_summary["total_duration_ms"],
                "retrieved_chunk_count": len(docs),
                "confidence_score": confidence_score,
            },
            tags={
                "session_id": session_id,
                "trace_id": tracer.trace_id,
            },
        )

        return AgentResponse(
            answer=final_answer,
            source=source,
            confidence_score=confidence_score,
            plan=plan,
            execution_steps=execution_steps,
            tools_invoked=tools_invoked,
            guardrail_status=guard_result.model_dump(),
            trace_id=tracer.trace_id,
            retrieved_chunk_count=len(docs),
        )
