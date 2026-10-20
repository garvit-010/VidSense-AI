import os
import re
from typing import Optional


class QueryRewriter:
    """
    High-Performance Search Query Optimizer.
    Converts verbose questions into keyword-dense retrieval queries with zero latency
    and zero dependence on brittle external API endpoints.
    """

    STOPWORDS = {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "the", "a", "an", "and", "or", "but", "in", "on",
        "at", "to", "for", "of", "with", "by", "from", "up", "about", "into",
        "over", "after", "can", "could", "would", "should", "did", "tell", "me",
        "explain", "describe", "according", "video", "mentioned", "please"
    }

    def __init__(self, *args, **kwargs):
        pass

    def rewrite(self, question: str) -> str:
        """Extracts high-salience search keywords from the user question."""
        cleaned = re.sub(r"[^\w\s-]", " ", question)
        tokens = cleaned.strip().split()
        keywords = [t for t in tokens if t.lower() not in self.STOPWORDS and len(t) > 2]

        if not keywords:
            return question.strip()

        # Keep top 8 meaningful search keywords
        return " ".join(keywords[:8])

    def rewrite_with_memory(self, question: str, memory: str) -> str:
        """Combines recent memory context entities with current query terms."""
        mem_tokens = [t for t in re.sub(r"[^\w\s-]", " ", memory).split() if len(t) > 3 and t.lower() not in self.STOPWORDS]
        q_tokens = [t for t in re.sub(r"[^\w\s-]", " ", question).split() if len(t) > 2 and t.lower() not in self.STOPWORDS]

        # Interleave context keywords
        combined = list(dict.fromkeys(q_tokens + mem_tokens[:4]))
        return " ".join(combined[:10])
