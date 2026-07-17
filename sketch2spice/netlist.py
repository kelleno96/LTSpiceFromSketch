"""Deterministic Circuit -> SPICE artifacts.

``to_netlist`` produces a plain SPICE ``.cir`` that LTSpice can simulate directly
(this is the dependable output). ``to_asc`` produces a best-effort LTSpice ``.asc``
schematic for opening/editing in the GUI -- it lays symbols out on a grid and wires
connectivity via net-name flags rather than routed wires.
"""

from __future__ import annotations

import re

from sketch2spice.model import KIND_PREFIX, Circuit, Component

# SI suffix multipliers understood by SPICE (case-insensitive; 'meg' before 'm').
_SI = {
    "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3,
    "k": 1e3, "meg": 1e6, "g": 1e9, "t": 1e12,
}


def parse_si(token: str) -> float | None:
    """Parse a SPICE value like '5m', '1k', '4.7u', '1meg' into a float."""
    m = re.fullmatch(r"\s*([0-9.eE+-]+)\s*(meg|[fpnumkgt])?\s*", token, re.IGNORECASE)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    suffix = (m.group(2) or "").lower()
    return val * _SI.get(suffix, 1.0)

# LTspice standard-library symbol name for each component kind.
KIND_SYMBOL: dict[str, str] = {
    "resistor": "res",
    "capacitor": "cap",
    "inductor": "ind",
    "voltage_source": "voltage",
    "current_source": "current",
    "diode": "diode",
}

# Pin offsets (dx, dy) from a symbol's placement origin, for R0 orientation, in
# LTspice's 16-unit grid. Order matches Component.nodes order. These are the
# offsets from the stock .asy symbols; validated against `LTspice -netlist`.
SYMBOL_PINS: dict[str, list[tuple[int, int]]] = {
    "res": [(16, 16), (16, 96)],
    "cap": [(16, 16), (16, 96)],
    "ind": [(16, 16), (16, 96)],
    "voltage": [(0, 16), (0, 96)],
    "current": [(0, 16), (0, 96)],
    "diode": [(0, 16), (0, 96)],
}


def _ground_map(circuit: Circuit) -> dict[str, str]:
    """Map the circuit's ground net onto SPICE node '0'.

    SPICE requires the ground reference to be node ``0``. If the sketch used a
    different name (``gnd``, ``vss``, ...) rewrite it; leave everything else alone.
    """
    if circuit.ground_node in ("", "0"):
        return {}
    return {circuit.ground_node: "0"}


def _node(name: str, gmap: dict[str, str]) -> str:
    return gmap.get(name, name)


def to_netlist(circuit: Circuit) -> str:
    """Render ``circuit`` as SPICE netlist text."""
    gmap = _ground_map(circuit)
    lines: list[str] = [f"* {circuit.title}"]

    diode_models: dict[str, None] = {}

    for comp in circuit.components:
        prefix = KIND_PREFIX[comp.kind]
        # Keep the sketch's designator if it already starts with the right prefix,
        # otherwise prefix it so the SPICE element type is unambiguous.
        ref = comp.ref if comp.ref[:1].upper() == prefix else f"{prefix}{comp.ref}"
        nodes = " ".join(_node(n, gmap) for n in comp.nodes)

        if comp.kind == "diode":
            model = comp.model or "Dgeneric"
            diode_models[model] = None
            lines.append(f"{ref} {nodes} {model}".rstrip())
        else:
            lines.append(f"{ref} {nodes} {comp.value}".rstrip())

    for model in diode_models:
        lines.append(f".model {model} D")

    lines.append(f".{circuit.analysis.type} {circuit.analysis.args}".rstrip())
    lines.append(".end")
    return "\n".join(lines) + "\n"


def to_ngspice(netlist_text: str) -> str:
    """Normalise an LTspice-dialect netlist so ngspice runs it unchanged.

    Two differences bite in practice:
    * LTspice writes independent-source waveforms as ``SINE(...)``; ngspice wants ``SIN(...)``.
    * LTspice accepts ``.tran <tstop>`` (one arg); ngspice needs ``.tran <tstep> <tstop>``.
    """
    out: list[str] = []
    for line in netlist_text.splitlines():
        stripped = line.strip()
        low = stripped.lower()

        # Source waveform keyword.
        line = re.sub(r"\bSINE\s*\(", "SIN(", line, flags=re.IGNORECASE)

        # .tran with a single argument -> add a step of tstop/50.
        if low.startswith(".tran"):
            parts = stripped.split()
            if len(parts) == 2:
                tstop = parse_si(parts[1])
                if tstop:
                    line = f".tran {tstop / 50:g} {tstop:g}"
        elif low.startswith(".backanno"):
            continue  # LTspice-only, unknown to ngspice
        out.append(line)
    return "\n".join(out) + "\n"


def _asc_symbol_block(comp: Component, x: int, y: int, gmap: dict[str, str]) -> list[str]:
    sym = KIND_SYMBOL[comp.kind]
    prefix = KIND_PREFIX[comp.kind]
    ref = comp.ref if comp.ref[:1].upper() == prefix else f"{prefix}{comp.ref}"

    block = [f"SYMBOL {sym} {x} {y} R0", f"SYMATTR InstName {ref}"]
    if comp.kind == "diode":
        block.append(f"SYMATTR Value {comp.model or 'Dgeneric'}")
    elif comp.value:
        block.append(f"SYMATTR Value {comp.value}")

    # Connectivity by net name: put a FLAG (net label) exactly on each pin. Two
    # pins with the same net name -> connected, no routing needed.
    pins = SYMBOL_PINS[sym]
    for (dx, dy), node in zip(pins, comp.nodes):
        net = _node(node, gmap)
        block.append(f"FLAG {x + dx} {y + dy} {net}")
    return block


def to_asc(circuit: Circuit) -> str:
    """Render ``circuit`` as a best-effort LTspice ``.asc`` schematic.

    Symbols are placed left-to-right on a grid; nets connect by shared FLAG labels.
    Good enough to open, view, and hand-edit; not a tidy hand-drawn layout.
    """
    gmap = _ground_map(circuit)
    n = len(circuit.components)

    spacing = 192
    x0, y0 = 96, 96
    width = max(880, x0 + n * spacing + 200)

    lines = ["Version 4", f"SHEET 1 {width} 680"]
    for i, comp in enumerate(circuit.components):
        lines.extend(_asc_symbol_block(comp, x0 + i * spacing, y0, gmap))

    # SPICE analysis directive as a schematic directive (leading '!').
    directive = f".{circuit.analysis.type} {circuit.analysis.args}".rstrip()
    lines.append(f"TEXT {x0} {y0 + 320} Left 2 !{directive}")
    return "\n".join(lines) + "\n"
