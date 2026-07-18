"""Vision backends: photo of a circuit -> validated ``Circuit``.

Two interchangeable implementations behind one ``extract`` call:

* ``LocalOpenAIBackend`` -- a local llama.cpp server (OpenAI-compatible, no key).
  The default; good enough to develop against.
* ``AnthropicBackend`` -- Claude's vision API, for higher accuracy.

Both return the same ``Circuit``, so nothing downstream depends on the choice.
"""

from __future__ import annotations

import base64
import importlib.resources
import json
import os
import re
from typing import Callable, Protocol

# Progress callback: receives (channel, text_delta) as the model responds.
# Channels: "thinking" (reasoning tokens, if the model emits them), "output"
# (the JSON answer as it streams), "status" (backend notes, e.g. retries).
DeltaCallback = Callable[[str, str], None]

# Generous generation budget for every model call. A thinking model can spend
# most of its tokens on reasoning_content before ever emitting the real answer
# ("Give this one pass of thought" in the prompt asks it not to, but a small
# budget turns that into a hard failure -- an empty response -- instead of just
# a slower one). The local model here supports up to a 262K context, so this
# is cheap headroom, not a real constraint.
MAX_TOKENS = 16384

from sketch2spice.model import Circuit, circuit_json_schema

# A small, verified-valid netlist (see examples/rc_lowpass.cir, checked against
# ngspice in tests) and the exact Circuit JSON it corresponds to, given to the
# vision models as a worked example. Kept side by side so they can't drift apart
# unnoticed -- if you touch one, update the other and re-run the netlist through
# ``simulate.run``.
_EXAMPLE_NETLIST = (
    importlib.resources.files("sketch2spice")
    .joinpath("examples/rc_lowpass.cir")
    .read_text()
    .strip()
)
_EXAMPLE_CIRCUIT_JSON = """\
{
  "title": "RC low-pass filter",
  "ground_node": "0",
  "components": [
    {"ref": "V1", "kind": "voltage_source", "nodes": ["in", "0"], "value": "SINE(0 5 1k)", "confidence": 1.0},
    {"ref": "R1", "kind": "resistor", "nodes": ["in", "out"], "value": "1k", "confidence": 1.0},
    {"ref": "C1", "kind": "capacitor", "nodes": ["out", "0"], "value": "159n", "confidence": 1.0}
  ],
  "analysis": {"type": "tran", "args": "5m"},
  "notes": ""
}\
"""

EXTRACTION_PROMPT = """\
You are reading a photograph of a hand-drawn electronic circuit schematic and \
converting it into a structured netlist.

Do not ruminate. Look at each component and wire once, commit to the single most \
likely reading, and move on. Never revisit a decision once you've made it. Your \
reasoning for the whole image should be a few sentences total, not a paragraph per \
component -- if you catch yourself writing "wait", "actually", "let me reconsider", \
"hmm", or re-deriving something you already decided, stop immediately, go with what \
you already have, and move to the next component. There is no reward for extra \
thinking here: a fast, reasonable guess beats an exhaustive one, and you only get \
one look at the image regardless of how long you deliberate.

When a wire or label is genuinely ambiguous, do not enumerate alternative \
interpretations or trace it back and forth repeatedly -- pick the simplest reading \
that fits basic circuit conventions, write one short clause about it in "notes", \
and move on. This applies especially to op-amp power rails: in this schema an \
opamp is always an ideal 3-terminal device (IN+, IN-, OUT) with no power-supply \
nodes. If you see supply rail labels near an op-amp (e.g. "+15V"/"-15V", \
"+20V"/"-20V"), do not try to model them, do not create voltage-source components \
for them, and do not spend any time puzzling over how they connect -- just ignore \
them for the netlist. Only capture them if they're a real voltage/current source \
with signal-carrying wires that reach non-power terminals.

The sketch was almost certainly drawn by hand by an undergraduate engineering \
student, not a machine. Expect the normal mess of a hand sketch: wobbly lines, \
inconsistent symbols, a wire that doesn't quite touch a terminal it's meant to \
connect to, a value scribbled and rewritten, or even a genuine circuit mistake on \
the student's part. Your job is to faithfully transcribe what is drawn, not to \
silently "fix" it into a circuit that makes more textbook sense -- if something \
looks wrong or inconsistent, transcribe it as drawn and flag it in "notes" rather \
than guessing what the student "must have meant."

Rules:
- Identify every component: resistors, capacitors, inductors, voltage sources, \
current sources, diodes, zener diodes (zener), LEDs (led), bipolar transistors (bjt), \
MOSFETs (mosfet), op-amps (opamp), and dependent sources -- voltage-controlled voltage \
source (vcvs), voltage-controlled current source (vccs), current-controlled current \
source (cccs), current-controlled voltage source (ccvs). Dependent sources are rare; \
only use them if the sketch clearly draws a diamond-shaped source symbol (as opposed \
to the circle used for an independent source).
- Assign each component a reference designator (R1, C1, V1, Q1, M1, U1, D1, E1, G1, \
F1, H1, ...) and read its value as written (e.g. "1k", "10u", "5V"). For a source, \
capture its full spec if given (e.g. "SINE(0 5 1k)" or "DC 5"); if only a number is \
shown, use it. For a zener, the value is its breakdown voltage (e.g. "5.1"). For a \
vcvs/vccs the value is its gain; for a cccs/ccvs the value is its gain and "model" is \
the reference designator of the voltage source whose current it senses (e.g. "V1").
- Trace the wires to determine which net (node) each terminal connects to. Give nets \
short names; label the ground/reference net "0". Two terminals joined by a wire share \
one net name.
- Terminal order in "nodes" depends on the kind: two-terminal parts (R, L, C, diode, \
zener, led, cccs, ccvs) give two nets; for a voltage/current source list the + \
terminal first; for a diode/zener/led list anode then cathode; for a bjt list \
collector, base, emitter; for a mosfet list drain, gate, source (and bulk if drawn); \
for an opamp list non-inverting input (+), inverting input (-), then output; for a \
vcvs/vccs list output+, output-, then the two control-sensing terminals.
- For a bjt set "subtype" to "npn" or "pnp"; for a mosfet set "subtype" to "nmos" or \
"pmos" (read the arrow/symbol direction, or guess npn/nmos if unclear and note it).
- Choose a sensible analysis: transient (".tran") for anything with a time-varying \
source, otherwise an operating point (".op"). Provide the analysis arguments.
- Put anything you could not read clearly, or any guess you made, into "notes", and \
set a low "confidence" on components you are unsure about.

Return ONLY a JSON object matching the provided schema. Do not add commentary.

For reference, here is a valid SPICE netlist for a simple circuit (a sine source \
driving an RC low-pass filter):

{example_netlist}

And the JSON you must return for that circuit looks like this:

{example_json}\
""".format(example_netlist=_EXAMPLE_NETLIST, example_json=_EXAMPLE_CIRCUIT_JSON)


EXPLAIN_PROMPT = """\
You are an electrical engineering teaching assistant. A student sketched and \
reviewed this circuit, titled "{title}":

{netlist}
{notes_block}
Explain it like a TA reviewing a lab report, briefly:
- Name the topology (e.g. "RC low-pass filter", "non-inverting op-amp amplifier", \
"common-emitter amplifier", "half-wave rectifier").
- Derive the expected behavior symbolically from the actual component values shown \
above -- gain, cutoff/corner frequency, time constant, bias point, whichever apply \
-- showing the formula and the resulting numeric prediction.
- State any simplifying assumptions (e.g. ideal op-amp, transistor in active region).
- Say what to look for in the simulation results to confirm it's working as expected \
(e.g. "V(out) should settle near X V" or "the -3dB point should land near Y Hz").

Keep it to a few short paragraphs or a bulleted list, not an exhaustive essay. Use \
markdown. Do not repeat the netlist back verbatim.\
"""


def _build_explain_prompt(circuit: Circuit, netlist_text: str) -> str:
    notes_block = f"\nNotes from parsing the sketch: {circuit.notes}\n" if circuit.notes else ""
    return EXPLAIN_PROMPT.format(
        title=circuit.title or "Circuit from sketch", netlist=netlist_text.strip(), notes_block=notes_block
    )


def _build_prompt(correction: str | None, previous: Circuit | None) -> str:
    """Base prompt, or a follow-up prompt asking the model to fix a specific miss.

    The image is sent again alongside this -- re-showing the model its own prior
    JSON plus what the user says is wrong is far more reliable than a blind retry.
    """
    if not correction:
        return EXTRACTION_PROMPT
    prev_json = previous.model_dump_json(indent=2) if previous is not None else "{}"
    return (
        EXTRACTION_PROMPT
        + f"""

You already analyzed this image once and returned:

{prev_json}

The user reviewed that result and says: "{correction}"

Look at the image again and return a corrected JSON object (matching the same \
schema) that fixes what the user pointed out. Keep everything else from your \
previous reading unless it's affected by the correction.\
"""
    )


class VisionBackend(Protocol):
    def extract(
        self,
        image_bytes: bytes,
        media_type: str,
        on_delta: DeltaCallback | None = None,
        correction: str | None = None,
        previous: Circuit | None = None,
    ) -> Circuit: ...

    def explain(
        self,
        circuit: Circuit,
        netlist_text: str,
        on_delta: DeltaCallback | None = None,
    ) -> str: ...


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
    if not text:
        raise ValueError(
            "the model returned an empty response. This usually means the image "
            "plus prompt overflowed the model's context window or the grammar "
            "constraint stalled generation -- try a smaller max image size, or "
            "check the server log."
        )
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

    def extract(
        self,
        image_bytes: bytes,
        media_type: str,
        on_delta: DeltaCallback | None = None,
        correction: str | None = None,
        previous: Circuit | None = None,
    ) -> Circuit:
        from openai import OpenAI

        emit = on_delta or (lambda channel, text: None)
        client = OpenAI(base_url=self.base_url, api_key="not-needed")
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:{media_type};base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_prompt(correction, previous)},
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
        for i, response_format in enumerate(formats):
            if i > 0:
                emit("status", f"\nRetrying with {response_format['type']} ({last_err})\n")
            try:
                stream = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    temperature=0,
                    max_tokens=MAX_TOKENS,
                    stream=True,
                )
                parts: list[str] = []
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    # llama.cpp puts thinking-model reasoning in reasoning_content.
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        emit("thinking", reasoning)
                    if delta.content:
                        parts.append(delta.content)
                        emit("output", delta.content)
                return _loads_circuit("".join(parts))
            except Exception as exc:  # noqa: BLE001 -- try the next format
                last_err = exc
        raise RuntimeError(f"local vision extraction failed: {last_err}")

    def explain(
        self,
        circuit: Circuit,
        netlist_text: str,
        on_delta: DeltaCallback | None = None,
    ) -> str:
        from openai import OpenAI

        emit = on_delta or (lambda channel, text: None)
        client = OpenAI(base_url=self.base_url, api_key="not-needed")
        messages = [{"role": "user", "content": _build_explain_prompt(circuit, netlist_text)}]

        # A thinking model can burn its whole token budget on reasoning_content
        # before emitting any real answer -- give it as much room as extraction.
        stream = client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.2, max_tokens=MAX_TOKENS, stream=True
        )
        parts: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                emit("thinking", reasoning)
            if delta.content:
                parts.append(delta.content)
                emit("output", delta.content)
        return "".join(parts)


class AnthropicBackend:
    """Claude vision API via the Anthropic SDK (needs ANTHROPIC_API_KEY)."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")

    def extract(
        self,
        image_bytes: bytes,
        media_type: str,
        on_delta: DeltaCallback | None = None,
        correction: str | None = None,
        previous: Circuit | None = None,
    ) -> Circuit:
        import anthropic

        client = anthropic.Anthropic()
        b64 = base64.b64encode(image_bytes).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_prompt(correction, previous)},
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
        ]

        if on_delta is not None:
            # Stream the raw text so the caller can show progress; the prompt
            # already demands JSON-only output, which _loads_circuit validates.
            parts: list[str] = []
            with client.messages.stream(
                model=self.model, max_tokens=MAX_TOKENS, messages=messages
            ) as stream:
                for text in stream.text_stream:
                    parts.append(text)
                    on_delta("output", text)
            return _loads_circuit("".join(parts))

        resp = client.messages.parse(
            model=self.model,
            max_tokens=MAX_TOKENS,
            messages=messages,
            output_format=Circuit,
        )
        return resp.parsed_output

    def explain(
        self,
        circuit: Circuit,
        netlist_text: str,
        on_delta: DeltaCallback | None = None,
    ) -> str:
        import anthropic

        client = anthropic.Anthropic()
        prompt = _build_explain_prompt(circuit, netlist_text)
        emit = on_delta or (lambda channel, text: None)
        parts: list[str] = []
        with client.messages.stream(
            model=self.model, max_tokens=MAX_TOKENS, messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                parts.append(text)
                emit("output", text)
        return "".join(parts)


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
    on_delta: DeltaCallback | None = None,
    correction: str | None = None,
    previous: Circuit | None = None,
) -> Circuit:
    if backend is None or isinstance(backend, str):
        backend = build_backend(backend)
    return backend.extract(
        image_bytes, media_type, on_delta=on_delta, correction=correction, previous=previous
    )


def explain_circuit(
    circuit: Circuit,
    netlist_text: str,
    backend: VisionBackend | str | None = None,
    on_delta: DeltaCallback | None = None,
) -> str:
    if backend is None or isinstance(backend, str):
        backend = build_backend(backend)
    return backend.explain(circuit, netlist_text, on_delta=on_delta)
