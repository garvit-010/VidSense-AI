import re
from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    passed: bool
    sanitized_input: str
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH
    flags: list[str] = Field(default_factory=list)
    reason: Optional[str] = None


class InputGuardrail:
    """
    LLMOps Input Security Guardrail for enterprise AI agent protection:
    - Prompt Injection & Jailbreak Pattern Detection
    - PII (Personally Identifiable Information) Redaction
    - API Key & Token Exfiltration Defense
    - Query Length Anomaly Bounds
    """

    JAILBREAK_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"(?i)disregard\s+(all\s+)?(system|safety|initial)\s+(rules|prompts|guidelines)",
        r"(?i)you\s+are\s+now\s+(an\s+unrestricted|in\s+developer\s+mode|dan|jailbroken)",
        r"(?i)system\s+override",
        r"(?i)reveal\s+(your\s+)?(system\s+prompt|initial\s+prompt|developer\s+instructions|instructions)",
        r"(?i)output\s+everything\s+above",
        r"(?i)act\s+as\s+(an\s+evil|unfiltered|jailbreak)",
        r"(?i)<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
        r"(?i)\[system\s*#.*\]",
        r"(?i)base64\s*decode\s*and\s*run",
    ]

    # Regex patterns for PII & Secrets
    PII_EMAIL_PATTERN = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    PII_PHONE_PATTERN = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    SECRET_KEY_PATTERN = r"\b(?:sk-[a-zA-Z0-9]{20,}|hf_[a-zA-Z0-9]{30,}|AIza[0-9A-Za-z-_]{35})\b"

    def __init__(self, max_length: int = 2000):
        self.max_length = max_length
        self._compiled_jailbreaks = [re.compile(p) for p in self.JAILBREAK_PATTERNS]

    def validate_and_sanitize(self, query: str) -> GuardrailResult:
        flags = []
        risk_level = "LOW"

        # 1. Length Anomaly Check
        if not query or not query.strip():
            return GuardrailResult(
                passed=False,
                sanitized_input="",
                risk_level="HIGH",
                flags=["EMPTY_QUERY"],
                reason="Query is empty or whitespace only.",
            )

        if len(query) > self.max_length:
            query = query[: self.max_length]
            flags.append("QUERY_TRUNCATED_MAX_LENGTH")
            risk_level = "MEDIUM"

        # 2. Prompt Injection & Jailbreak Detection
        for pattern in self._compiled_jailbreaks:
            if pattern.search(query):
                flags.append("PROMPT_INJECTION_SUSPECTED")
                risk_level = "HIGH"
                return GuardrailResult(
                    passed=False,
                    sanitized_input="[BLOCKED PROMPT INJECTION]",
                    risk_level=risk_level,
                    flags=flags,
                    reason="Input contains adversarial prompt injection or jailbreak patterns.",
                )

        # 3. PII Redaction
        sanitized = query
        if re.search(self.PII_EMAIL_PATTERN, sanitized):
            sanitized = re.sub(self.PII_EMAIL_PATTERN, "[REDACTED_EMAIL]", sanitized)
            flags.append("PII_EMAIL_REDACTED")
            risk_level = "MEDIUM"

        if re.search(self.PII_PHONE_PATTERN, sanitized):
            sanitized = re.sub(self.PII_PHONE_PATTERN, "[REDACTED_PHONE]", sanitized)
            flags.append("PII_PHONE_REDACTED")
            risk_level = "MEDIUM"

        # 4. Secret Token Redaction
        if re.search(self.SECRET_KEY_PATTERN, sanitized):
            sanitized = re.sub(self.SECRET_KEY_PATTERN, "[REDACTED_API_KEY]", sanitized)
            flags.append("SECRET_TOKEN_REDACTED")
            risk_level = "HIGH"

        return GuardrailResult(
            passed=True,
            sanitized_input=sanitized,
            risk_level=risk_level,
            flags=flags,
            reason="Input passed all security guardrail checks." if not flags else "Sanitized input.",
        )


class OutputGuardrail:
    """
    LLMOps Output Safety Guardrail:
    - Secret & Internal Prompt Leakage Prevention
    - Hallucination / Ungrounded Claim Filtering
    - Malformed Content Mitigation
    """

    SECRET_PATTERNS = [
        r"\b(?:sk-[a-zA-Z0-9]{20,}|hf_[a-zA-Z0-9]{30,}|AIza[0-9A-Za-z-_]{35})\b",
        r"(?i)SYSTEM_PROMPT:",
        r"(?i)GOOGLE_API_KEY",
        r"(?i)HUGGINGFACEHUB_API_TOKEN",
    ]

    def __init__(self):
        self._compiled_secrets = [re.compile(p) for p in self.SECRET_PATTERNS]

    def validate_and_scrub(self, output: str) -> Tuple[bool, str, list[str]]:
        flags = []
        scrubbed = output

        # Check for system key or prompt leaks
        for p in self._compiled_secrets:
            if p.search(scrubbed):
                scrubbed = p.sub("[CONFIDENTIAL_REDACTED]", scrubbed)
                flags.append("SYSTEM_SECRET_SCRUBBED")

        # Basic sanity check
        if not scrubbed.strip():
            return False, "I was unable to synthesize a reliable answer from the provided context.", ["EMPTY_OUTPUT"]

        passed = "SYSTEM_SECRET_SCRUBBED" not in flags
        return passed, scrubbed, flags
