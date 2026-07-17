"""Streamlit app: photo of a hand-drawn circuit -> reviewed netlist -> LTspice sim."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sketch2spice import simulate
from sketch2spice.model import Analysis, Circuit, Component
from sketch2spice.netlist import to_asc, to_netlist
from sketch2spice.viz import render_schematic
from sketch2spice.vision import extract_circuit

COMPONENT_KINDS = [
    "resistor",
    "capacitor",
    "inductor",
    "voltage_source",
    "current_source",
    "diode",
]
ANALYSIS_KINDS = ["tran", "ac", "op", "dc"]

st.set_page_config(page_title="Sketch -> LTspice", layout="wide")
st.title("Hand-drawn circuit -> LTspice simulation")


def circuit_to_rows(circuit: Circuit) -> list[dict]:
    rows = []
    for c in circuit.components:
        nodes = list(c.nodes) + ["", ""]
        rows.append(
            {
                "ref": c.ref,
                "kind": c.kind,
                "node+": nodes[0],
                "node-": nodes[1],
                "value": c.value,
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
        comps.append(
            Component(
                ref=ref,
                kind=kind,
                nodes=[_cell(r, "node+"), _cell(r, "node-")],
                value=_cell(r, "value"),
                model=_cell(r, "model") or None,
                confidence=min(1.0, max(0.0, conf)),
            )
        )
    return comps


# ---- Sidebar: backend selection --------------------------------------------
with st.sidebar:
    st.header("Vision backend")
    backend_choice = st.radio(
        "Model used to read the sketch",
        options=["local", "anthropic"],
        format_func=lambda x: {
            "local": "Local (llama.cpp @ :8080)",
            "anthropic": "Claude API (needs key)",
        }[x],
        index=0,
    )
    st.caption(
        "Local runs against http://127.0.0.1:8080 with no API key. "
        "Claude needs ANTHROPIC_API_KEY set in the environment."
    )

# ---- Step 1: upload & analyze ----------------------------------------------
uploaded = st.file_uploader(
    "Photo of the circuit sketch", type=["png", "jpg", "jpeg", "webp"]
)

col_img, col_do = st.columns([2, 1])
if uploaded is not None:
    col_img.image(uploaded, caption=uploaded.name, use_container_width=True)
    if col_do.button("Analyze sketch", type="primary"):
        with st.spinner("Reading the sketch..."):
            try:
                circuit = extract_circuit(
                    uploaded.getvalue(),
                    uploaded.type or "image/png",
                    backend=backend_choice,
                )
                st.session_state.circuit = circuit
                st.session_state.netlist_text = to_netlist(circuit)
            except Exception as exc:  # surfaced to the user, not swallowed
                st.error(f"Could not read the sketch: {exc}")

# ---- Step 2: human-in-the-loop review --------------------------------------
circuit: Circuit | None = st.session_state.get("circuit")
if circuit is not None:
    st.subheader("Review the parsed circuit")
    if circuit.notes:
        st.warning(f"Model notes: {circuit.notes}")

    meta1, meta2, meta3, meta4 = st.columns(4)
    title = meta1.text_input("Title", circuit.title)
    ground = meta2.text_input("Ground net", circuit.ground_node)
    a_type = meta3.selectbox(
        "Analysis", ANALYSIS_KINDS, index=ANALYSIS_KINDS.index(circuit.analysis.type)
    )
    a_args = meta4.text_input("Analysis args", circuit.analysis.args)

    st.caption(
        "Edit components below. Two-terminal parts only; use '0' for ground. "
        "For sources, node+ is the positive terminal."
    )
    edited = st.data_editor(
        pd.DataFrame(circuit_to_rows(circuit)),
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "kind": st.column_config.SelectboxColumn("kind", options=COMPONENT_KINDS),
            "confidence": st.column_config.NumberColumn(
                "confidence", min_value=0.0, max_value=1.0, step=0.05, format="%.2f"
            ),
        },
        key="component_editor",
    )

    reviewed = Circuit(
        title=title,
        ground_node=ground,
        components=rows_to_components(edited),
        analysis=Analysis(type=a_type, args=a_args),
        notes=circuit.notes,
    )

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

    # ---- Step 4: simulate & plot -------------------------------------------
    st.subheader("Simulate")
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
                fig, ax = plt.subplots()
                for n in selected:
                    ax.plot(result.x, result.traces[n], label=n)
                ax.set_xlabel(result.x_name)
                ax.set_ylabel("value")
                ax.legend(loc="best", fontsize="small")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        with st.expander("LTspice log"):
            st.code(result.log or "(empty)")
