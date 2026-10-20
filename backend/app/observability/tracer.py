import time
import uuid
from typing import Dict, Any, List, Optional
from contextlib import contextmanager
from app.utils.logger import log_event


class Span:
    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.metadata = metadata or {}
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration_ms: float = 0.0
        self.error: Optional[str] = None

    def start(self):
        self.start_time = time.perf_counter()
        return self

    def finish(self, error: Optional[str] = None):
        self.end_time = time.perf_counter()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        if error:
            self.error = str(error)

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "name": self.name,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }
        if self.error:
            res["error"] = self.error
        return res


class AgentTracer:
    """
    LLMOps Distributed Tracing & Performance Profiler.
    Collects granular span latencies, memory footprint, tool invocations,
    and publishes structured telemetry records for monitoring dashboards.
    """

    def __init__(self, session_id: Optional[str] = None, trace_id: Optional[str] = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.session_id = session_id or "default_session"
        self.start_time = time.perf_counter()
        self.spans: List[Span] = []
        self.attributes: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
        }

    @contextmanager
    def span(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        current_span = Span(name=name, metadata=metadata)
        current_span.start()
        try:
            yield current_span
            current_span.finish()
        except Exception as e:
            current_span.finish(error=str(e))
            raise
        finally:
            self.spans.append(current_span)

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def finish(self) -> Dict[str, Any]:
        total_duration_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        trace_summary = {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "total_duration_ms": total_duration_ms,
            "spans": [s.to_dict() for s in self.spans],
            "attributes": self.attributes,
        }
        # Log to structured event log
        log_event({"event_type": "llmops_trace", **trace_summary})
        return trace_summary
