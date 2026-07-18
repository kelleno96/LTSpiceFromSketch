"""Streamlit app: photo of a hand-drawn circuit -> reviewed netlist -> LTspice sim."""

from __future__ import annotations

import os
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

from sketch2spice import erc, simulate
from sketch2spice.model import Analysis, Circuit, Component
from sketch2spice.netlist import parse_si, to_asc, to_netlist
from sketch2spice.viz import render_schematic
from sketch2spice.vision import LocalOpenAIBackend, explain_circuit, extract_circuit

DEFAULT_VISION_URL = os.environ.get("LOCAL_VISION_URL", "http://127.0.0.1:8080/v1")

COMPONENT_KINDS = [
    "resistor",
    "capacitor",
    "inductor",
    "voltage_source",
    "current_source",
    "diode",
    "zener",
    "led",
    "bjt",
    "mosfet",
    "opamp",
    "vcvs",
    "vccs",
    "cccs",
    "ccvs",
]
SUBTYPES = ["", "npn", "pnp", "nmos", "pmos"]
ANALYSIS_KINDS = ["tran", "ac", "op", "dc"]

# One-click "how do I drive this?" presets. Each fills the input source's value
# with a concrete SPICE waveform *and* sets the matching analysis + args, since
# the two have to agree (a bare "AC" source does nothing in a .tran, and .ac
# ignores a SINE(...) waveform). Everything lands in editable fields, so these
# are just sensible starting points the student can tweak.
STIMULUS_PRESETS: dict[str, dict[str, str]] = {
    "AC sweep → Bode": {
        "value": "AC 1",
        "type": "ac",
        "args": "dec 20 0.1 1meg",
        "help": "Frequency response: gain/phase vs. frequency. 1 V AC stimulus, "
        "swept 0.1 Hz–1 MHz. Widen the range in Analysis args if your corner "
        "frequency falls outside it.",
    },
    "Sine → transient": {
        "value": "SINE(0 1 1k)",
        "type": "tran",
        "args": "5m",
        "help": "Time-domain response to a 1 V, 1 kHz sine (5 ms window ≈ 5 cycles).",
    },
    "Pulse/step → transient": {
        "value": "PULSE(0 1 0 10u 10u 2m 4m)",
        "type": "tran",
        "args": "10m",
        "help": "Step/square response: 0→1 V pulses, 2 ms high every 4 ms.",
    },
    "DC level → bias": {
        "value": "DC 1",
        "type": "op",
        "args": "",
        "help": "Hold the input at a fixed 1 V DC and solve the operating point.",
    },
}

st.set_page_config(page_title="Sketch -> LTspice", layout="wide")
st.title("Hand-drawn circuit -> LTspice simulation")


def circuit_to_rows(circuit: Circuit) -> list[dict]:
    rows = []
    for c in circuit.components:
        nodes = list(c.nodes) + ["", "", "", ""]
        rows.append(
            {
                "ref": c.ref,
                "kind": c.kind,
                "n1": nodes[0],
                "n2": nodes[1],
                "n3": nodes[2],
                "n4": nodes[3],
                "value": c.value,
                "subtype": c.subtype or "",
                "model": c.model or "",
                "confidence": c.confidence,
            }
        )
    return rows


def _cell(row, key: str, default: str = "") -> str:
    """Read a data_editor cell as a clean string (empty for NaN/None)."""
    val = row.get(key)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return str(val).strip()


def rows_to_components(df: pd.DataFrame) -> list[Component]:
    comps = []
    for _, r in df.iterrows():
        ref = _cell(r, "ref")
        kind = _cell(r, "kind")
        if not ref or not kind:
            continue  # skip blank / half-filled rows
        conf_raw = r.get("confidence")
        conf = 1.0 if conf_raw is None or (isinstance(conf_raw, float) and pd.isna(conf_raw)) else float(conf_raw)
        nodes = [_cell(r, "n1"), _cell(r, "n2"), _cell(r, "n3"), _cell(r, "n4")]
        while nodes and nodes[-1] == "":  # drop unused trailing terminals
            nodes.pop()
        comps.append(
            Component(
                ref=ref,
                kind=kind,
                nodes=nodes,
                value=_cell(r, "value"),
                model=_cell(r, "model") or None,
                subtype=_cell(r, "subtype") or None,
                confidence=min(1.0, max(0.0, conf)),
            )
        )
    return comps


_INPUT_NET_NAMES = {"in", "vin", "input", "source", "sig", "signal", "vs"}


def _pick_input_net(circuit: Circuit, ground: str) -> str | None:
    """Best guess at the net an input source should drive, when none was drawn.

    Prefers a net named like an input (``vin``/``in``/...); failing that, a
    floating net (touched by exactly one terminal), which is almost always where
    a missing source belongs on a hand sketch. Returns None if nothing fits.
    """
    counts: dict[str, int] = {}
    for c in circuit.components:
        for n in c.nodes:
            if n:
                counts[n] = counts.get(n, 0) + 1
    nongnd = [n for n in counts if n not in (ground, "0")]
    named = [n for n in nongnd if n.lower() in _INPUT_NET_NAMES or n.lower().startswith(("in", "vin"))]
    if named:
        return named[0]
    floating = [n for n in nongnd if counts[n] == 1]
    return floating[0] if floating else None


def _next_source_ref(circuit: Circuit) -> str:
    existing = {c.ref.upper() for c in circuit.components}
    i = 1
    while f"V{i}" in existing:
        i += 1
    return f"V{i}"


def apply_stimulus_preset(reviewed: Circuit, preset: dict[str, str]) -> tuple[str, str]:
    """Set the input source's waveform + the analysis to a preset, preserving edits.

    Works on ``reviewed`` (which already folds in any manual table edits), stores
    the result as the new base circuit, and bumps ``form_ver`` so the analysis
    widgets and the data editor remount and re-seed from it. If the circuit has no
    voltage source (the input wasn't drawn/parsed), synthesizes one driving the
    best-guess input net. Returns ``(severity, message)`` for a flash notice.
    """
    new = reviewed.model_copy(deep=True)
    new.analysis = Analysis(type=preset["type"], args=preset["args"])
    analysis_desc = f"{preset['type']} {preset['args']}".strip()

    target = next((c for c in new.components if c.kind == "voltage_source"), None)
    if target is not None:
        target.value = preset["value"]
        flash = ("success", f"Set {target.ref} = {preset['value']}, analysis = {analysis_desc}.")
    else:
        ground = new.ground_node or "0"
        net = _pick_input_net(new, ground)
        if net is not None:
            ref = _next_source_ref(new)
            new.components.insert(
                0, Component(ref=ref, kind="voltage_source", nodes=[net, ground], value=preset["value"])
            )
            flash = (
                "success",
                f"No source was drawn — added {ref} = {preset['value']} driving net "
                f"'{net}' (referenced to {ground}), analysis = {analysis_desc}. "
                "Rewire it in the table if that's not your input.",
            )
        else:
            flash = (
                "warning",
                f"Set analysis to {analysis_desc}, but there's no voltage source and "
                "no obvious input net to add one to — add a row like "
                f"V1, voltage_source, <input net>, 0, {preset['value']}.",
            )

    st.session_state.circuit = new
    st.session_state.form_ver = st.session_state.get("form_ver", 0) + 1
    # Keep the netlist textarea in sync so "Run LTspice" uses the preset directly.
    st.session_state.netlist_text = to_netlist(new)
    return flash


def downscale_image(data: bytes, media_type: str, max_dim: int) -> tuple[bytes, str, tuple[int, int]]:
    """Shrink the photo so its longest side is <= max_dim (0 disables).

    Phone photos are often 4000px+, which blows up token usage (and can overflow
    a local model's context entirely); the sketch survives a downscale fine.
    """
    img = Image.open(BytesIO(data))
    if max_dim <= 0 or max(img.size) <= max_dim:
        return data, media_type, img.size
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = BytesIO()
    if img.mode in ("RGBA", "LA", "P"):
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png", img.size
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue(), "image/jpeg", img.size


def _find_minus_3db(freq: np.ndarray, mag: np.ndarray) -> tuple[float, float] | None:
    """First frequency where ``mag`` crosses 3 dB below its own peak.

    Interpolates linearly in log-frequency / dB space between the two samples
    that bracket the crossing. Returns None if the trace never drops that far
    (e.g. a flat passband with no rolloff in the swept range).
    """
    if len(freq) < 2 or np.max(mag) <= 0:
        return None
    mag_db = 20 * np.log10(np.maximum(mag, 1e-30))
    target = np.max(mag_db) - 3.0
    for i in range(1, len(freq)):
        a, b = mag_db[i - 1], mag_db[i]
        if a >= target > b:
            frac = (a - target) / (a - b) if a != b else 0.0
            log_f = np.log10(freq[i - 1]) + frac * (np.log10(freq[i]) - np.log10(freq[i - 1]))
            return 10**log_f, target
    return None


def bode_figure(result: "simulate.SimResult", names: list[str]) -> go.Figure:
    """Stacked magnitude (dB) / phase (deg) Bode plot with -3 dB markers."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for n in names:
        mag = result.traces[n]
        mag_db = 20 * np.log10(np.maximum(mag, 1e-30))
        fig.add_trace(go.Scatter(x=result.x, y=mag_db, name=n, mode="lines"), row=1, col=1)

        cplx = result.complex_traces.get(n)
        phase = np.degrees(np.angle(cplx)) if cplx is not None else np.zeros_like(mag)
        fig.add_trace(
            go.Scatter(x=result.x, y=phase, name=n, mode="lines", legendgroup=n, showlegend=False),
            row=2,
            col=1,
        )

        cross = _find_minus_3db(result.x, mag)
        if cross is not None:
            f0, y0 = cross
            fig.add_trace(
                go.Scatter(
                    x=[f0],
                    y=[y0],
                    mode="markers",
                    marker=dict(symbol="x", size=9, color="red"),
                    name=f"{n} -3dB",
                    hovertemplate=f"{n} -3dB @ %{{x:.4g}} Hz<extra></extra>",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

    fig.update_xaxes(type="log", title_text="frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="magnitude (dB)", row=1, col=1)
    fig.update_yaxes(title_text="phase (deg)", row=2, col=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    return fig


def linear_figure(result: "simulate.SimResult", names: list[str]) -> go.Figure:
    fig = go.Figure()
    for n in names:
        fig.add_trace(go.Scatter(x=result.x, y=result.traces[n], name=n, mode="lines"))
    fig.update_layout(
        xaxis_title=result.x_name,
        yaxis_title="value",
        height=480,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h"),
    )
    return fig


_ENG_SUFFIXES = [
    (1e12, "T"), (1e9, "G"), (1e6, "Meg"), (1e3, "k"),
    (1.0, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p"), (1e-15, "f"),
]


def format_engineering(value: float, unit: str = "") -> str:
    """Render a float in SPICE-style engineering notation, e.g. 0.0025 -> '2.5m'."""
    if not np.isfinite(value):
        return str(value)
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v == 0:
        return f"0{unit}"
    for scale, suffix in _ENG_SUFFIXES:
        if v >= scale:
            return f"{sign}{v / scale:.4g}{suffix}{unit}"
    return f"{sign}{v:.4g}{unit}"


def operating_point_tables(result: "simulate.SimResult") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a ``.op`` result's single-point values into voltage/current tables.

    ngspice's raw file for a sweepless analysis has no real x axis, so
    ``simulate.run`` treats the first named value as ``x`` -- fold it back in
    with the rest so a node like the supply rail isn't silently dropped.
    """
    values = {result.x_name: float(result.x[0])}
    values.update({n: float(result.traces[n][0]) for n in result.signal_names()})

    volt_rows = [
        {"node": n, "voltage": format_engineering(v, "V")}
        for n, v in values.items()
        if n.lower().startswith("v(")
    ]
    curr_rows = [
        {"device": n, "current": format_engineering(v, "A")}
        for n, v in values.items()
        if n.lower().startswith("i(")
    ]
    return pd.DataFrame(volt_rows), pd.DataFrame(curr_rows)


def make_backend(backend_choice: str, vision_url: str):
    """Resolve the sidebar's backend choice into what extract_circuit/explain_circuit expect."""
    return LocalOpenAIBackend(base_url=vision_url.strip()) if backend_choice == "local" else backend_choice


def run_extraction(
    img_bytes: bytes,
    img_type: str,
    img_size: tuple[int, int],
    backend_choice: str,
    vision_url: str,
    correction: str | None = None,
    previous: Circuit | None = None,
) -> bool:
    """Stream a vision extraction into a live status panel; results go to session_state.

    Shared by the initial "Analyze sketch" button and the "Re-analyze with
    correction" flow in the review section -- both just supply different
    ``correction``/``previous`` arguments over the same stored image.
    """
    label = "Re-analyzing with your correction..." if correction else "Reading the sketch..."
    with st.status(label, expanded=True) as status:
        st.caption(
            f"Sending {img_size[0]}x{img_size[1]} image "
            f"({len(img_bytes) / 1024:.0f} KB) to the {backend_choice} backend."
        )
        if correction:
            st.caption(f"Correction: {correction}")
        status_ph = st.empty()
        thinking_ph = st.empty()
        output_ph = st.empty()
        buffers = {"status": "", "thinking": "", "output": ""}
        placeholders = {"status": status_ph, "thinking": thinking_ph, "output": output_ph}

        def on_delta(channel: str, text: str) -> None:
            buffers[channel] += text
            if channel == "status":
                placeholders[channel].caption(buffers[channel])
            else:
                lbl = "Thinking" if channel == "thinking" else "Model output"
                placeholders[channel].code(f"{lbl}:\n{buffers[channel]}", language=None)

        try:
            backend = make_backend(backend_choice, vision_url)
            circuit = extract_circuit(
                img_bytes,
                img_type,
                backend=backend,
                on_delta=on_delta,
                correction=correction,
                previous=previous,
            )
            st.session_state.circuit = circuit
            st.session_state.netlist_text = to_netlist(circuit)
            st.session_state.last_model_output = dict(buffers)
            st.session_state.last_image = (img_bytes, img_type, img_size)
            status.update(label="Sketch parsed", state="complete", expanded=False)
            return True
        except Exception as exc:  # surfaced to the user, not swallowed
            st.session_state.last_model_output = dict(buffers)
            status.update(label="Sketch analysis failed", state="error", expanded=True)
            st.error(f"Could not read the sketch: {exc}")
            return False


def run_explain(circuit: Circuit, netlist_text: str, backend_choice: str, vision_url: str) -> None:
    """Stream a plain-text (no image) explanation of the reviewed circuit."""
    with st.status("Thinking through the circuit...", expanded=True) as status:
        output_ph = st.empty()
        buffers = {"thinking": "", "output": ""}

        def on_delta(channel: str, text: str) -> None:
            if channel not in buffers:
                return
            buffers[channel] += text
            output_ph.markdown(buffers["output"] or "_...thinking..._")

        try:
            backend = make_backend(backend_choice, vision_url)
            explanation = explain_circuit(circuit, netlist_text, backend=backend, on_delta=on_delta)
            st.session_state.explanation = explanation
            status.update(label="Explanation ready", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Explanation failed", state="error", expanded=True)
            st.error(f"Could not explain the circuit: {exc}")


def traces_to_csv(result: "simulate.SimResult", names: list[str]) -> str:
    data = {result.x_name: result.x}
    for n in names:
        data[n] = result.traces[n]
        if n in result.complex_traces:
            data[f"{n}_phase_deg"] = np.degrees(np.angle(result.complex_traces[n]))
    return pd.DataFrame(data).to_csv(index=False)


# ---- Sidebar: backend selection --------------------------------------------
with st.sidebar:
    st.header("Vision backend")
    backend_choice = st.radio(
        "Model used to read the sketch",
        options=["local", "anthropic"],
        format_func=lambda x: {
            "local": "Local (OpenAI-compatible)",
            "anthropic": "Claude API (needs key)",
        }[x],
        index=0,
    )
    vision_url = DEFAULT_VISION_URL
    if backend_choice == "local":
        vision_url = st.text_input(
            "Local model endpoint",
            value=DEFAULT_VISION_URL,
            help="OpenAI-compatible base URL — include the trailing /v1, "
            "e.g. http://127.0.0.1:8080/v1. Defaults to LOCAL_VISION_URL / --vision-url.",
        )
    st.caption(
        "Local runs against the endpoint above with no API key. "
        "Claude needs ANTHROPIC_API_KEY set in the environment."
    )

    st.header("Image preprocessing")
    max_dim = st.number_input(
        "Max image dimension (px)",
        min_value=0,
        max_value=8192,
        value=1024,
        step=128,
        help="Downscale the photo before sending it to the model so the longest "
        "side is at most this many pixels. 0 disables downscaling. Large phone "
        "photos can overflow a local model's context window.",
    )

# ---- Step 1: upload & analyze ----------------------------------------------
uploaded = st.file_uploader(
    "Photo of the circuit sketch", type=["png", "jpg", "jpeg", "webp"]
)

col_img, col_do = st.columns([2, 1])
if uploaded is not None:
    col_img.image(uploaded, caption=uploaded.name, use_container_width=True)
    if col_do.button("Analyze sketch", type="primary"):
        img_bytes, img_type, img_size = downscale_image(
            uploaded.getvalue(), uploaded.type or "image/png", int(max_dim)
        )
        run_extraction(img_bytes, img_type, img_size, backend_choice, vision_url)

# Raw output survives reruns (any widget interaction) in this expander.
last_out = st.session_state.get("last_model_output")
if last_out and (last_out["thinking"] or last_out["output"]):
    with st.expander("Model output from the last analysis"):
        if last_out["thinking"]:
            st.markdown("**Thinking**")
            st.code(last_out["thinking"], language=None)
        st.markdown("**Output**")
        st.code(last_out["output"] or "(empty)", language="json")

# ---- Step 2: human-in-the-loop review --------------------------------------
circuit: Circuit | None = st.session_state.get("circuit")
if circuit is not None:
    st.subheader("Review the parsed circuit")
    if circuit.notes:
        st.warning(f"Model notes: {circuit.notes}")

    # Bumping form_ver remounts the analysis widgets + data editor so they re-seed
    # from st.session_state.circuit (used by the stimulus presets below).
    form_ver = st.session_state.setdefault("form_ver", 0)
    # Rendered later (after `reviewed` exists) but positioned here, above the table.
    stimulus_slot = st.container()

    meta1, meta2, meta3, meta4 = st.columns(4)
    title = meta1.text_input("Title", circuit.title)
    ground = meta2.text_input("Ground net", circuit.ground_node)
    a_type = meta3.selectbox(
        "Analysis", ANALYSIS_KINDS, index=ANALYSIS_KINDS.index(circuit.analysis.type),
        key=f"analysis_type_{form_ver}",
    )
    a_args = meta4.text_input("Analysis args", circuit.analysis.args, key=f"analysis_args_{form_ver}")

    st.caption(
        "Edit components below; use '0' for ground. Terminal order (n1…n4) by kind — "
        "**R/L/C:** n1, n2 · **source:** +, − (n1 = +) · **diode/zener/led:** anode, "
        "cathode · **bjt:** C, B, E · **mosfet:** D, G, S, (bulk) · **opamp:** IN+, "
        "IN−, OUT · **vcvs/vccs:** OUT+, OUT−, CTRL+, CTRL− · **cccs/ccvs:** OUT+, "
        "OUT− (put the sensed source's ref, e.g. 'V1', in **model**). "
        "Set **subtype** (npn/pnp/nmos/pmos) for transistors. Zener value = "
        "breakdown voltage; vcvs/vccs/cccs/ccvs value = gain."
    )
    edited = st.data_editor(
        pd.DataFrame(circuit_to_rows(circuit)),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "kind": st.column_config.SelectboxColumn("kind", options=COMPONENT_KINDS),
            "subtype": st.column_config.SelectboxColumn("subtype", options=SUBTYPES, required=False),
            "confidence": st.column_config.NumberColumn(
                "confidence", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"
            ),
        },
        key=f"component_editor_{form_ver}",
    )

    reviewed = Circuit(
        title=title,
        ground_node=ground,
        components=rows_to_components(edited),
        analysis=Analysis(type=a_type, args=a_args),
        notes=circuit.notes,
    )

    # ---- Stimulus presets: one click sets the source waveform + analysis ----
    with stimulus_slot:
        st.caption(
            "**Not sure how to drive it?** Pick an input stimulus — this fills the "
            "source's value and the analysis settings (which have to match) with an "
            "editable starting point. Applied to the first voltage source."
        )
        preset_cols = st.columns(len(STIMULUS_PRESETS))
        for col, (name, preset) in zip(preset_cols, STIMULUS_PRESETS.items()):
            if col.button(name, help=preset["help"], use_container_width=True):
                st.session_state.stimulus_flash = apply_stimulus_preset(reviewed, preset)
                st.rerun()
        flash = st.session_state.pop("stimulus_flash", None)
        if flash:
            (st.success if flash[0] == "success" else st.warning)(flash[1])

    # ---- Electrical rule check: catch wiring/value mistakes early ----------
    for finding in erc.check(reviewed):
        ref_suffix = f" ({finding.ref})" if finding.ref else ""
        (st.error if finding.severity == "error" else st.warning)(f"{finding.message}{ref_suffix}")

    # ---- Correction loop: send the photo back with what the model missed ---
    last_image = st.session_state.get("last_image")
    if last_image is not None:
        with st.expander("Tell the model what it misread"):
            st.caption(
                "Faster than hand-editing every cell: describe the mistake and "
                "the model re-reads the same photo with your note and its own "
                "prior answer in front of it."
            )
            note = st.text_area(
                "What's wrong",
                placeholder="e.g. \"R2 is 4.7k, not 47k\" or \"the diode is flipped, cathode is on the left\"",
                key="correction_note",
            )
            if st.button("Re-analyze with correction"):
                if not note.strip():
                    st.warning("Describe what's wrong first.")
                else:
                    img_bytes, img_type, img_size = last_image
                    if run_extraction(
                        img_bytes, img_type, img_size, backend_choice, vision_url,
                        correction=note.strip(), previous=reviewed,
                    ):
                        st.rerun()

    # ---- Schematic vs photo: does the parse match the drawing? -------------
    st.subheader("Parsed schematic vs. photo")
    st.caption(
        "Compare the redrawn schematic (right) against your sketch (left) to check "
        "the components and wiring were read correctly. Low-confidence reads are "
        "labelled in red."
    )
    view_photo, view_schem = st.columns(2)
    if uploaded is not None:
        view_photo.image(uploaded, caption="Your sketch", use_container_width=True)
    if reviewed.components:
        try:
            view_schem.image(
                render_schematic(reviewed), caption="Parsed schematic", use_container_width=True
            )
        except Exception as exc:  # unusual topology — table stays authoritative
            view_schem.warning(f"Couldn't lay out a schematic for this circuit ({exc}).")
    else:
        view_schem.info("No components yet.")

    # ---- Step 3: netlist (editable escape hatch) + downloads ---------------
    st.subheader("SPICE netlist")
    if st.button("Regenerate netlist from table"):
        st.session_state.netlist_text = to_netlist(reviewed)
    st.session_state.setdefault("netlist_text", to_netlist(reviewed))
    st.text_area("Netlist (.cir) — edit if needed before simulating", key="netlist_text", height=220)

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Download .cir netlist",
        st.session_state.netlist_text,
        file_name="circuit.cir",
        mime="text/plain",
    )
    dl2.download_button(
        "Download .asc schematic",
        to_asc(reviewed),
        file_name="circuit.asc",
        mime="text/plain",
    )

    # ---- Explain this circuit (text-only LLM call, no image) ---------------
    st.subheader("Explain this circuit")
    if st.button("Explain this circuit"):
        run_explain(reviewed, st.session_state.netlist_text, backend_choice, vision_url)

    explanation = st.session_state.get("explanation")
    if explanation:
        with st.expander("Explanation", expanded=True):
            st.markdown(explanation)

    # ---- Step 4: simulate & plot -------------------------------------------
    st.subheader("Simulate")

    with st.expander("Bias point (DC operating point)"):
        st.caption(
            "Runs a separate .op analysis (your Simulate settings above are untouched) "
            "and reports every node voltage and source/device current ngspice computes."
        )
        if st.button("Compute operating point"):
            op_netlist = to_netlist(
                Circuit(
                    title=reviewed.title,
                    ground_node=reviewed.ground_node,
                    components=reviewed.components,
                    analysis=Analysis(type="op", args=""),
                    notes=reviewed.notes,
                )
            )
            try:
                st.session_state.op_result = simulate.run(op_netlist)
            except Exception as exc:
                st.session_state.pop("op_result", None)
                st.error(f"Operating point failed: {exc}")

        op_result = st.session_state.get("op_result")
        if op_result is not None:
            volt_df, curr_df = operating_point_tables(op_result)
            col_v, col_i = st.columns(2)
            col_v.markdown("**Node voltages**")
            if not volt_df.empty:
                col_v.dataframe(volt_df, use_container_width=True, hide_index=True)
            else:
                col_v.caption("No node voltages reported.")
            col_i.markdown("**Currents**")
            if not curr_df.empty:
                col_i.dataframe(curr_df, use_container_width=True, hide_index=True)
            else:
                col_i.caption("No currents reported.")

    if st.button("Run LTspice", type="primary"):
        with st.spinner("Running LTspice..."):
            try:
                result = simulate.run(st.session_state.netlist_text)
                st.session_state.sim = result
            except Exception as exc:
                st.session_state.pop("sim", None)
                st.error(f"Simulation failed: {exc}")

    result = st.session_state.get("sim")
    if result is not None:
        names = result.signal_names()
        if not names:
            st.info("The simulation produced no plottable traces.")
        elif len(result.x) <= 1:
            st.write({n: result.traces[n].tolist() for n in names})
        else:
            default = [n for n in names if n.upper().startswith("V(")] or names
            selected = st.multiselect("Signals to plot", names, default=default[:6])
            if selected:
                if result.is_ac:
                    st.caption(
                        "AC sweep: Bode plot (magnitude in dB, phase in degrees). "
                        "Red x marks each trace's -3 dB point."
                    )
                    st.plotly_chart(bode_figure(result, selected), use_container_width=True)
                else:
                    st.plotly_chart(linear_figure(result, selected), use_container_width=True)
                st.download_button(
                    "Download traces (.csv)",
                    traces_to_csv(result, selected),
                    file_name="traces.csv",
                    mime="text/csv",
                )
        with st.expander("LTspice log"):
            st.code(result.log or "(empty)")

    with st.expander("Parameter sweep"):
        st.caption(
            "Step one component's value across a range and overlay the resulting "
            "traces -- e.g. see how a filter's cutoff or an amplifier's gain shifts "
            "with a part value."
        )
        sweepable = [c.ref for c in reviewed.components if parse_si(c.value) is not None]
        if not sweepable:
            st.caption("No components with a plain numeric value (e.g. '1k') to sweep.")
        else:
            sw_ref = st.selectbox("Component", sweepable, key="sweep_ref")
            sw1, sw2, sw3, sw4 = st.columns(4)
            start_str = sw1.text_input("Start", value="1k", key="sweep_start")
            stop_str = sw2.text_input("Stop", value="100k", key="sweep_stop")
            n_points = sw3.number_input("Points", min_value=2, max_value=15, value=5, key="sweep_points")
            spacing = sw4.selectbox("Spacing", ["log", "linear"], key="sweep_spacing")

            if st.button("Run sweep"):
                start, stop = parse_si(start_str), parse_si(stop_str)
                if start is None or stop is None:
                    st.error("Start/stop must be numbers ngspice understands, e.g. '1k', '4.7u'.")
                elif spacing == "log" and (start <= 0 or stop <= 0):
                    st.error("Log spacing needs positive start/stop values.")
                else:
                    values = (
                        np.geomspace(start, stop, int(n_points))
                        if spacing == "log"
                        else np.linspace(start, stop, int(n_points))
                    )
                    progress = st.progress(0.0)
                    runs: list[tuple[float, simulate.SimResult]] = []
                    errors: list[tuple[float, str]] = []
                    for i, val in enumerate(values):
                        swept = reviewed.model_copy(deep=True)
                        for c in swept.components:
                            if c.ref == sw_ref:
                                c.value = format_engineering(float(val))
                        try:
                            runs.append((float(val), simulate.run(to_netlist(swept))))
                        except Exception as exc:
                            errors.append((float(val), str(exc)))
                        progress.progress((i + 1) / len(values))
                    progress.empty()
                    st.session_state.sweep_runs = runs
                    st.session_state.sweep_errors = errors
                    st.session_state.sweep_swept_ref = sw_ref

            runs = st.session_state.get("sweep_runs")
            if runs:
                all_names = sorted({n for _, r in runs for n in r.signal_names()})
                if all_names:
                    default_sig = next((n for n in all_names if n.upper().startswith("V(")), all_names[0])
                    sig = st.selectbox(
                        "Signal to plot", all_names, index=all_names.index(default_sig), key="sweep_signal"
                    )
                    swept_ref = st.session_state.get("sweep_swept_ref", "")
                    fig = go.Figure()
                    for val, r in runs:
                        if sig in r.traces:
                            fig.add_trace(
                                go.Scatter(
                                    x=r.x, y=r.traces[sig], mode="lines",
                                    name=f"{swept_ref}={format_engineering(val)}",
                                )
                            )
                    fig.update_layout(
                        xaxis_title=runs[0][1].x_name,
                        yaxis_title=sig,
                        height=480,
                        margin=dict(l=10, r=10, t=30, b=10),
                        legend=dict(orientation="h"),
                    )
                    if runs[0][1].is_ac:
                        fig.update_xaxes(type="log")
                    st.plotly_chart(fig, use_container_width=True)
                errors = st.session_state.get("sweep_errors") or []
                if errors:
                    st.warning(
                        f"{len(errors)} sweep point(s) failed to simulate: "
                        + ", ".join(format_engineering(v) for v, _ in errors)
                    )
