"""LLM clients shared by the goal-extraction pipelines.

Qwen served locally by LM Studio generates and reconstructs goals. Ministral
evaluates top-down and repair candidates; Ministral, Granite and Gemma form
the bottom-up goal-review ensemble. No conversation history is retained.
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from threading import RLock
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel


QWEN_GENERATOR_MODEL = os.getenv("QWEN_GENERATOR_MODEL", "qwen/qwen3-vl-8b")
QWEN_GENERATOR_MAX_TOKENS = int(os.getenv("QWEN_GENERATOR_MAX_TOKENS", "8192"))
LM_STUDIO_GENERATOR_ATTEMPTS = int(os.getenv("LM_STUDIO_GENERATOR_ATTEMPTS", "3"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "600"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "1"))
if not QWEN_GENERATOR_MODEL.strip() or LM_STUDIO_GENERATOR_ATTEMPTS < 1:
    raise ValueError("A local Qwen model and a positive attempt budget are required")

LM_STUDIO_BASE_URL = os.getenv(
    "LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"
).rstrip("/")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
LM_STUDIO_MANAGEMENT_BASE_URL = os.getenv(
    "LM_STUDIO_MANAGEMENT_BASE_URL",
    LM_STUDIO_BASE_URL.removesuffix("/v1") + "/api/v1",
).rstrip("/")
LM_STUDIO_MODEL_MANAGEMENT_TIMEOUT_SECONDS = float(
    os.getenv("LM_STUDIO_MODEL_MANAGEMENT_TIMEOUT_SECONDS", "600")
)
LM_STUDIO_EVALUATOR_CONTEXT_LENGTH = int(
    os.getenv("LM_STUDIO_EVALUATOR_CONTEXT_LENGTH", "16384")
)
MINISTRAL_CRITIC_MODEL = os.getenv(
    "MINISTRAL_CRITIC_MODEL", "mistralai/ministral-3-14b-reasoning"
).strip()
GRANITE_CRITIC_MODEL = os.getenv(
    "GRANITE_CRITIC_MODEL", "ibm/granite-3.2-8b"
).strip()
GEMMA_CRITIC_MODEL = os.getenv(
    "GEMMA_CRITIC_MODEL", "google/gemma-4-12b-qat"
).strip()
MINISTRAL_CRITIC_MAX_TOKENS = int(
    os.getenv("MINISTRAL_CRITIC_MAX_TOKENS", "4096")
)
BOTTOM_UP_CRITIC_MAX_TOKENS = int(
    os.getenv("BOTTOM_UP_CRITIC_MAX_TOKENS", "4096")
)
qwen_client = None
ministral_client = None
_resolved_ministral_model: str | None = None


def _qwen_client():
    global qwen_client
    if qwen_client is None:
        from openai import OpenAI
        qwen_client = OpenAI(api_key=LM_STUDIO_API_KEY, base_url=LM_STUDIO_BASE_URL,
                             timeout=REQUEST_TIMEOUT_SECONDS, max_retries=API_MAX_RETRIES)
    return qwen_client


def reset_model_clients():
    """Close the lazily created local API connections."""
    global qwen_client, ministral_client
    for client in (qwen_client, ministral_client):
        if client is not None:
            client.close()
    qwen_client = ministral_client = None


def _extract_json(text: str) -> Any:
    """Decode the first JSON value, tolerating fences and reasoning tags."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("Qwen did not return a valid JSON object.")


def generate_response(
    prompt: str, sys_prompt: str, response_format: type[BaseModel] | None = None,
    *, stateless_retries: bool = False,
) -> str | BaseModel:
    """Use fresh messages per call; optionally repeat invalid JSON without history."""
    system_content = sys_prompt
    if response_format is not None:
        system_content += ("\nReturn only one JSON object matching this schema:\n"
                           + json.dumps(response_format.model_json_schema(), ensure_ascii=False))
    messages = [{"role": "system", "content": system_content}, {"role": "user", "content": prompt}]
    last_exception = None
    with loaded_local_model(QWEN_GENERATOR_MODEL):
        for _ in range(LM_STUDIO_GENERATOR_ATTEMPTS):
            try:
                response = _qwen_client().chat.completions.create(
                    model=QWEN_GENERATOR_MODEL, messages=messages, temperature=0,
                    max_tokens=QWEN_GENERATOR_MAX_TOKENS)
            except Exception as exc:
                raise RuntimeError(f"Qwen locale {QWEN_GENERATOR_MODEL!r} su {LM_STUDIO_BASE_URL}: {exc}") from exc
            choice = response.choices[0]
            content = choice.message.content
            if choice.finish_reason == "length":
                raise RuntimeError("Qwen reached its output-token budget; no incomplete result was accepted. Increase QWEN_GENERATOR_MAX_TOKENS and the loaded context.")
            if not content:
                raise RuntimeError("Local Qwen returned no final response; verify reasoning and output-token budgets.")
            if response_format is None:
                return content
            try:
                return response_format.model_validate(_extract_json(content))
            except ValueError as exc:
                last_exception = exc
                if not stateless_retries:
                    messages.extend([{"role": "assistant", "content": content},
                                     {"role": "user", "content": f"Correct the complete JSON to match the schema. Validation error: {exc}"}])
    raise ValueError(f"Local Qwen produced invalid structured output after {LM_STUDIO_GENERATOR_ATTEMPTS} attempts") from last_exception


def _ministral_client():
    global ministral_client
    if ministral_client is None:
        from openai import OpenAI

        ministral_client = OpenAI(
            api_key=LM_STUDIO_API_KEY,
            base_url=LM_STUDIO_BASE_URL,
            timeout=REQUEST_TIMEOUT_SECONDS, max_retries=API_MAX_RETRIES,
        )
    return ministral_client


def _ministral_model() -> str:
    """Resolve an explicit or loaded Ministral model from LM Studio."""
    global _resolved_ministral_model
    if MINISTRAL_CRITIC_MODEL:
        return MINISTRAL_CRITIC_MODEL
    if _resolved_ministral_model:
        return _resolved_ministral_model

    models = list(_ministral_client().models.list().data)
    ids = [model.id for model in models]
    matches = [model_id for model_id in ids if "ministral" in model_id.lower()]
    if not matches:
        loaded = ", ".join(ids) if ids else "nessuno"
        raise RuntimeError(
            "Nessun modello Ministral caricato in LM Studio. Carica Ministral "
            "oppure imposta MINISTRAL_CRITIC_MODEL con il model id esatto. "
            f"Modelli rilevati: {loaded}."
        )
    _resolved_ministral_model = matches[0]
    return _resolved_ministral_model


def configured_critic_model() -> str:
    """Return the configured model id without contacting LM Studio when possible."""
    return MINISTRAL_CRITIC_MODEL or _resolved_ministral_model or "auto:ministral"


def configured_bottom_up_critic_models() -> tuple[str, str, str]:
    """Return the heterogeneous three-model bottom-up critic ensemble."""
    return (
        configured_critic_model(),
        GRANITE_CRITIC_MODEL,
        GEMMA_CRITIC_MODEL,
    )


def _lm_studio_management_request(
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call LM Studio's native model-management REST API."""
    url = f"{LM_STUDIO_MANAGEMENT_BASE_URL}/{path.lstrip('/')}"
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if LM_STUDIO_API_KEY:
        headers["Authorization"] = f"Bearer {LM_STUDIO_API_KEY}"
    request = Request(
        url,
        data=body,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    try:
        with urlopen(
            request,
            timeout=LM_STUDIO_MODEL_MANAGEMENT_TIMEOUT_SECONDS,
        ) as response:
            content = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"LM Studio model-management request failed for {url}: {exc}"
        ) from exc
    return json.loads(content) if content else {}


def _loaded_lm_studio_instance_configs() -> dict[str, list[dict]]:
    payload = _lm_studio_management_request("models")
    loaded = {}
    for model in payload.get("models", []):
        model_id = str(model.get("key", "")).strip()
        instances = [instance for instance in model.get("loaded_instances", [])
                     if str(instance.get("id", "")).strip()]
        if model_id and instances:
            loaded[model_id] = instances
    return loaded


def _loaded_lm_studio_instances() -> dict[str, list[str]]:
    return {model: [str(instance["id"]).strip() for instance in instances]
            for model, instances in _loaded_lm_studio_instance_configs().items()}


_local_load_configs: dict[str, dict] = {}


def _unload_lm_studio_instance(instance_id: str) -> None:
    _lm_studio_management_request(
        "models/unload",
        {"instance_id": instance_id},
    )


def _unload_other_bottom_up_critics(selected_model: str) -> None:
    ensemble = {*configured_bottom_up_critic_models(), QWEN_GENERATOR_MODEL}
    for model_id, instance_ids in _loaded_lm_studio_instances().items():
        if model_id not in ensemble or model_id == selected_model:
            continue
        for instance_id in instance_ids:
            _unload_lm_studio_instance(instance_id)


@contextmanager
def _loaded_local_model_instance(model_id: str, *, context_length: int | None = None):
    """Load one local pipeline model and unload it at the end of the lease."""
    selected_model = model_id.strip()
    if not selected_model:
        raise ValueError("The LM Studio evaluator model id cannot be empty.")

    _unload_other_bottom_up_critics(selected_model)
    loaded = _loaded_lm_studio_instance_configs()
    existing_instances = loaded.get(selected_model, [])
    required_context = LM_STUDIO_EVALUATOR_CONTEXT_LENGTH if context_length is None else context_length
    if required_context < 1:
        raise ValueError("Requested model context must be positive")
    suitable = [instance for instance in existing_instances
                if isinstance((instance.get("config") or {}).get("context_length"), int)
                and instance["config"]["context_length"] >= required_context]
    instance_id = str(suitable[0]["id"]).strip() if suitable else None
    for instance in existing_instances:
        if str(instance["id"]).strip() != instance_id:
            _unload_lm_studio_instance(str(instance["id"]).strip())
    if instance_id:
        load_config = suitable[0]["config"]
    else:
        response = _lm_studio_management_request(
            "models/load",
            {
                "model": selected_model,
                "context_length": required_context,
                "flash_attention": True,
                "offload_kv_cache_to_gpu": True,
                "echo_load_config": True,
            },
        )
        instance_id = str(
            response.get("instance_id")
            or response.get("model_instance_id")
            or ""
        ).strip()
        if not instance_id:
            raise RuntimeError(
                f"LM Studio loaded {selected_model!r} but returned no instance id."
            )
        load_config = response.get("load_config") or {}
        if not isinstance(load_config.get("context_length"), int):
            instances = _loaded_lm_studio_instance_configs().get(selected_model, [])
            load_config = next((instance.get("config") or {} for instance in instances
                                if str(instance["id"]).strip() == instance_id), {})

    actual_context = load_config.get("context_length")
    if not isinstance(actual_context, int) or actual_context < required_context:
        _unload_lm_studio_instance(instance_id)
        raise RuntimeError(f"LM Studio context verification failed for {selected_model}: "
                           f"requested={required_context}, actual={actual_context}. "
                           "No silent context reduction is allowed.")
    _local_load_configs[selected_model] = {
        "loaded_instance_id": instance_id, "loaded_context_length": actual_context,
        "requested_context_length": required_context,
    }

    try:
        yield selected_model
    finally:
        try:
            _unload_lm_studio_instance(instance_id)
        finally:
            _local_load_configs.pop(selected_model, None)


_model_lock = RLock()
_active_local_model = None


@contextmanager
def loaded_local_model(model_id: str, *, context_length: int | None = None):
    """One GPU model per process; nested leases of the same model reuse it."""
    global _active_local_model
    with _model_lock:
        if _active_local_model is not None:
            if _active_local_model != model_id:
                raise RuntimeError(f"Cannot load {model_id} while {_active_local_model} has an active lease")
            if context_length is not None and _local_load_configs.get(model_id, {}).get("loaded_context_length", 0) < context_length:
                raise RuntimeError("Cannot enlarge context inside an active nested model lease")
            yield model_id
            return
        with _loaded_local_model_instance(model_id, context_length=context_length) as selected:
            _active_local_model = selected
            try:
                yield selected
            finally:
                _active_local_model = None


loaded_local_evaluator_model = loaded_local_model


class EvaluatorResponse(str):
    """Critic text with request and generation diagnostics."""

    def __new__(cls, content, metadata):
        result = super().__new__(cls, content)
        result.metadata = metadata
        return result


class EvaluatorGenerationError(RuntimeError):
    def __init__(self, message, raw_response, metadata):
        super().__init__(message)
        self.raw_response = raw_response
        self.metadata = metadata


def _generate_evaluator_response(
    prompt: str,
    sys_prompt: str,
    *,
    temperature: float = 0,
    model: str | None = None,
    response_schema: dict | None = None,
) -> EvaluatorResponse:
    """One stateless chat-completions request, textual or schema-constrained.

    The top-down critic uses text; bottom-up evaluators supply a JSON schema.
    No native chat endpoint or free-text fallback is used for structured calls.
    """
    selected_model = model.strip() if model else _ministral_model()
    if not selected_model:
        raise ValueError("The LM Studio evaluator model id cannot be empty.")
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]
    structured = response_schema is not None
    max_tokens = BOTTOM_UP_CRITIC_MAX_TOKENS if structured else MINISTRAL_CRITIC_MAX_TOKENS
    metadata = {
        "endpoint": "/v1/chat/completions", "structured_output": structured,
        "model": selected_model, "temperature": temperature,
        "max_output_tokens": max_tokens,
        "prompt_characters": sum(len(message["content"]) for message in messages),
        **_local_load_configs.get(selected_model, {}),
    }
    options = {}
    if structured:
        options["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "documentary_review", "strict": True, "schema": response_schema}}
    try:
        response = _ministral_client().chat.completions.create(
            model=selected_model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, **options)
    except Exception as exc:
        raise EvaluatorGenerationError(
            f"Local critic {selected_model!r} on {LM_STUDIO_BASE_URL}: {exc}",
            "", metadata) from exc
    choice = response.choices[0]
    content = choice.message.content or ""
    metadata.update(
        finish_reason=choice.finish_reason,
        usage=response.usage.model_dump(mode="json") if response.usage is not None else None)
    if choice.finish_reason != "stop" or not content.strip():
        raise EvaluatorGenerationError(
            f"Critic response incomplete or unavailable (finish_reason={choice.finish_reason!r}). "
            "Check output-token and loaded-context budgets; no partial response accepted.",
            content, metadata)
    return EvaluatorResponse(content, metadata)


def generate_evaluator_response(
    prompt: str,
    sys_prompt: str,
    *,
    temperature: float = 0,
    model: str | None = None,
    response_schema: dict | None = None,
) -> EvaluatorResponse:
    """Load the selected local model and make one request without chat history."""
    selected = model.strip() if model else _ministral_model()
    with loaded_local_model(selected):
        return _generate_evaluator_response(
            prompt, sys_prompt, temperature=temperature, model=selected,
            response_schema=response_schema)
