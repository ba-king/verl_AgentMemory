from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .schemas import EvidencePacket


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_text(self, text: str) -> list[float]: ...


def _require_header_safe(value: str, name: str) -> None:
    try:
        value.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            f"{name} contains non-latin characters. Please use the real API key value, "
            "not a localized placeholder."
        ) from exc


@dataclass
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float = 60.0

    @classmethod
    def from_env(
        cls,
        *,
        prefix: str,
        fallback_prefix: str = "OPENAI",
        default_base_url: str = "https://api.openai.com/v1",
    ) -> "OpenAICompatibleConfig":
        base_url = (
            os.environ.get(f"{prefix}_BASE_URL")
            or os.environ.get(f"{fallback_prefix}_BASE_URL")
            or default_base_url
        )
        api_key = os.environ.get(f"{prefix}_API_KEY") or os.environ.get(f"{fallback_prefix}_API_KEY")
        model = os.environ.get(f"{prefix}_MODEL") or os.environ.get(f"{fallback_prefix}_MODEL")
        if not api_key:
            raise RuntimeError(f"Set {prefix}_API_KEY or {fallback_prefix}_API_KEY.")
        if not model:
            raise RuntimeError(f"Set {prefix}_MODEL or {fallback_prefix}_MODEL.")
        timeout = _timeout_from_env(prefix=prefix, fallback_prefix=fallback_prefix, default=60.0)
        return cls(base_url=base_url, api_key=api_key, model=model, timeout=timeout)


def _timeout_from_env(*, prefix: str, fallback_prefix: str, default: float) -> float:
    raw = os.environ.get(f"{prefix}_TIMEOUT") or os.environ.get(f"{fallback_prefix}_TIMEOUT")
    if raw is None:
        return default
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{prefix}_TIMEOUT must be a number of seconds, got: {raw}") from exc
    if timeout <= 0:
        raise RuntimeError(f"{prefix}_TIMEOUT must be positive, got: {raw}")
    return timeout


class OpenAICompatibleChatClient:
    """Small stdlib client for OpenAI-compatible chat completions.

    Works with providers exposing `/chat/completions`, including OpenAI,
    DeepSeek-compatible endpoints, and local vLLM/SGLang servers.
    """

    def __init__(self, config: OpenAICompatibleConfig):
        _require_header_safe(config.api_key, "api_key")
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    @classmethod
    def from_env(cls, *, prefix: str, fallback_prefix: str = "OPENAI") -> "OpenAICompatibleChatClient":
        return cls(OpenAICompatibleConfig.from_env(prefix=prefix, fallback_prefix=fallback_prefix))

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        response_format: dict[str, str] | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_payload:
            payload.update(extra_payload)
        body = self._post_json("/chat/completions", payload)
        return body["choices"][0]["message"]["content"]

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        extra_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        text = self.chat(
            messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            extra_payload=extra_payload,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Model did not return valid JSON: {text}") from exc

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Chat completion failed with HTTP {exc.code}: {detail}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Chat completion timed out after {self.config.timeout:.0f}s. "
                f"Increase {self.config.model} timeout with P2R_MAIN_TIMEOUT/P2R_MEMORY_TIMEOUT or OPENAI_TIMEOUT."
            ) from exc


class OpenAICompatibleEmbeddingClient:
    """Small stdlib client for OpenAI-compatible embedding APIs."""

    def __init__(self, config: OpenAICompatibleConfig):
        _require_header_safe(config.api_key, "api_key")
        self.config = config
        self.base_url = config.base_url.rstrip("/")

    @classmethod
    def from_env(cls, *, prefix: str = "P2R_EMBEDDING", fallback_prefix: str = "OPENAI") -> "OpenAICompatibleEmbeddingClient":
        return cls(OpenAICompatibleConfig.from_env(prefix=prefix, fallback_prefix=fallback_prefix))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.config.model, "input": texts}
        body = self._post_json("/embeddings", payload)
        data = sorted(body["data"], key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in data]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Embedding request failed with HTTP {exc.code}: {detail}") from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Embedding request timed out after {self.config.timeout:.0f}s. "
                "Increase the timeout with P2R_EMBEDDING_TIMEOUT or OPENAI_TIMEOUT."
            ) from exc


class LocalQwenEmbeddingClient:
    """Local Qwen3 embedding client using transformers + torch.

    Default checkpoint path:
        ~/.cache/p2r_cmc/models/Qwen3-Embedding-0.6B

    Override with:
        P2R_LOCAL_EMBEDDING_MODEL_PATH=/path/to/Qwen3-Embedding-0.6B
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_path = Path(
            model_path
            or os.environ.get("P2R_LOCAL_EMBEDDING_MODEL_PATH")
            or (Path.home() / ".cache" / "p2r_cmc" / "models" / "Qwen3-Embedding-0.6B")
        ).expanduser()
        self.device = device or os.environ.get("P2R_LOCAL_EMBEDDING_DEVICE") or "cpu"
        self.max_length = max_length
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @classmethod
    def from_env(cls) -> "LocalQwenEmbeddingClient":
        return cls()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        tokenizer, model, torch = self._load()
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, return_dict=True)
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs.hidden_states[-1]
            embeddings = self._mean_pool(hidden, encoded["attention_mask"], torch)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().tolist()

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _load(self) -> tuple[Any, Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            import torch

            return self._tokenizer, self._model, torch
        if not self.model_path.exists():
            raise RuntimeError(f"Local Qwen embedding checkpoint not found: {self.model_path}")
        try:
            import torch
            from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LocalQwenEmbeddingClient requires torch and transformers. "
                "Install them with: python -m pip install --user torch transformers"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        try:
            self._model = AutoModel.from_pretrained(str(self.model_path), trust_remote_code=True)
        except Exception:
            self._model = AutoModelForCausalLM.from_pretrained(str(self.model_path), trust_remote_code=True)
        self._model.to(self.device)
        self._model.eval()
        return self._tokenizer, self._model, torch

    @staticmethod
    def _mean_pool(hidden: Any, attention_mask: Any, torch: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        summed = torch.sum(hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts


class MainAnswerAgent:
    """Main answering agent. It receives query plus EvidencePacket."""

    def __init__(self, chat_client: OpenAICompatibleChatClient):
        self.chat_client = chat_client

    @classmethod
    def from_env(cls) -> "MainAnswerAgent":
        return cls(OpenAICompatibleChatClient.from_env(prefix="P2R_MAIN"))

    def answer(self, query: str, evidence_packet: EvidencePacket) -> str:
        evidence_lines = []
        for idx, item in enumerate(evidence_packet.evidence_items, start=1):
            facets = ", ".join(item.facets) if item.facets else "none"
            evidence_lines.append(
                "\n".join(
                    [
                        f"{idx}. memory_id={item.memory_id}",
                        f"   summary: {item.summary}",
                        f"   content: {item.content}",
                        f"   facets: {facets}",
                    ]
                )
            )
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "<none>"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the main agent. Answer the user query using the provided memory evidence "
                    "only when it is relevant. If evidence is insufficient, say so briefly.\n\n"
                    "Evidence-use rules:\n"
                    "1. Treat explicit preferences, constraints, allergies, dietary restrictions, mobility limits, "
                    "deadlines, and safety requirements as hard constraints.\n"
                    "2. Treat explicit requirements in the current user query as hard constraints too, especially "
                    "numbers, dates, counts, budgets, ordered steps, required sections, output formats, scope limits, "
                    "and alternatives marked with words like 'or', 'either', 'choose one', '二选一', or '或'.\n"
                    "3. Preserve hard constraints exactly. Do not replace a specific constraint with a vague "
                    "generalization. If evidence contains a specific restriction, threshold, exception, or named item, "
                    "keep that detail explicit whenever it affects the answer.\n"
                    "4. Respect exclusivity and cardinality. If the user asks for one of several alternatives, choose "
                    "one or clearly compare them without executing all of them. If the user asks for exactly N items, "
                    "sections, days, steps, examples, or actions, provide exactly N unless you explain why not.\n"
                    "5. When planning, cover every relevant evidence item at least once, unless it is clearly "
                    "irrelevant to the user query.\n"
                    "6. If two evidence items conflict, trust the verified evidence packet and do not use omitted "
                    "or contradicted memories.\n"
                    "7. Before finalizing the answer, silently check that the plan satisfies both the current query "
                    "constraints and the memory constraints. If there is a tradeoff, briefly explain it.\n"
                    "8. Do not expose memory IDs in the final answer unless the user asks for debugging details."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Verified memory evidence:\n{evidence_text}\n\n"
                    f"Verifier notes:\n{json.dumps(evidence_packet.conflicts, ensure_ascii=False)}\n\n"
                    f"User query:\n{query}"
                ),
            },
        ]
        return self.chat_client.chat(messages, temperature=0.2)
