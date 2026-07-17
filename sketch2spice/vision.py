"""Vision backends: photo of a circuit -> validated ``Circuit``.

Two interchangeable implementations behind one ``extract`` call:

* ``LocalOpenAIBackend`` -- a local llama.cpp server (OpenAI-compatible, no key).
  The default; good enough to develop against.
* ``AnthropicBackend`` -- Claude's vision API, for higher accuracy.

Both return the same ``Circuit``, so nothing downstream depends on the choice.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Protocol

from sketch2spice.model import Circuit, circuit_json_schema

EXTRACTION_PROMPT = """\
You are reading a photograph of a hand-drawn electronic circuit schematic and \
converting it into a structured netlist.

Rules:
- Identify every component: resistors, capacitors, inductors, voltage sources, \
current sources, and diodes.
- Assign each component a reference designator (R1, C1, V1, ...) and read its value \
as written (e.g. "1k", "10u", "5V"). For a source, capture its full spec if given \
(e.g. "SINE(0 5 1k)" or "DC 5"); if only a number is shown, use it.
- Trace the wires to determine which net (node) each terminal connects to. Give nets \
short names; label the ground/reference net "0". Two terminals joined by a wire share \
one net name.
- For a voltage or current source, list the + terminal's net first.
- Choose a sensible analysis: transient (".tran") for anything with a time-varying \
source, otherwise an operating point (".op"). Provide the analysis arguments.
- Put anything you could not read clearly, or any guess you made, into "notes", and \
set a low "confidence" on components you are unsure about.

Return ONLY a JSON object matching the provided schema. Do not add commentary.\
"""


class VisionBackend(Protocol):
    def extract(self, image_bytes: bytes, media_type: str) -> Circuit: ...


def _deref_schema(schema: dict) -> dict:
    """Inline ``$ref``/``$defs`` into a self-contained schema.

    llama.cpp builds a GBNF grammar from the ``response_format`` schema and handles
    a flat, dereferenced schema far more reliably than one with ``$ref`` pointers.
    """
    defs = schema.get("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                return resolve(dict(defs.get(name, {})))
            return {k: resolve(v) for k, v in node.items() if k != "$defs"}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def _loads_circuit(text: str) -> Circuit:
    """Validate model output into a ``Circuit``, tolerating fenced/extra text."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return Circuit.model_validate_json(text)
    except Exception:
        # Fall back to grabbing the outermost JSON object.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return Circuit.model_validate(json.loads(match.group(0)))


class LocalOpenAIBackend:
    """OpenAI-compatible local server (llama.cpp) with a vision model."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = base_url or os.environ.get(
            "LOCAL_VISION_URL", "http://127.0.0.1:8080/v1"
        )
        self.model = model or os.environ.get("LOCAL_VISION_MODEL", "serve-rpc")

    def extract(self, image_bytes: bytes, media_type: str) -> Circuit:
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key="not-needed")
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:{media_type};base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

        schema = _deref_schema(circuit_json_schema())
        # Prefer grammar-constrained JSON (json_schema); fall back to a plain
        # json_object if the server rejects the schema *or* returns something that
        # doesn't parse (some models truncate under strict grammar constraints).
        formats = [
            {"type": "json_schema", "json_schema": {"name": "circuit", "schema": schema}},
            {"type": "json_object"},
        ]
        last_err: Exception | None = None
        for response_format in formats:
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    temperature=0,
                    max_tokens=4096,
                )
                return _loads_circuit(resp.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001 -- try the next format
                last_err = exc
        raise RuntimeError(f"local vision extraction failed: {last_err}")


class AnthropicBackend:
    """Claude vision API via the Anthropic SDK (needs ANTHROPIC_API_KEY)."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")

    def extract(self, image_bytes: bytes, media_type: str) -> Circuit:
        import anthropic

        client = anthropic.Anthropic()
        b64 = base64.b64encode(image_bytes).decode("ascii")

        resp = client.messages.parse(
            model=self.model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                    ],
                }
            ],
            output_format=Circuit,
        )
        return resp.parsed_output


def build_backend(name: str | None = None) -> VisionBackend:
    """Construct a backend by name (env ``VISION_BACKEND``, default 'local')."""
    name = (name or os.environ.get("VISION_BACKEND", "local")).lower()
    if name in ("local", "llamacpp", "openai"):
        return LocalOpenAIBackend()
    if name in ("anthropic", "claude"):
        return AnthropicBackend()
    raise ValueError(f"unknown vision backend: {name!r}")


def extract_circuit(
    image_bytes: bytes,
    media_type: str,
    backend: VisionBackend | str | None = None,
) -> Circuit:
    if backend is None or isinstance(backend, str):
        backend = build_backend(backend)
    return backend.extract(image_bytes, media_type)
