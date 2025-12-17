from collections import defaultdict, deque


class SessionMemory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.store = defaultdict(lambda: deque(maxlen=max_turns))

    def add(self, session_id: str, role: str, content: str):
        self.store[session_id].append(
            {"role": role, "content": content}
        )

    def get_context(self, session_id: str) -> str:
        history = self.store.get(session_id, [])
        return "\n".join(
            f"{m['role']}: {m['content']}" for m in history
        )
