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

# LTspice standard-library symbol name for each component kind. Kinds with no
# entry (multi-terminal actives, dependent sources) have no simple 2-pin stock
# symbol to place on the grid layout below, so to_asc skips them with a comment
# rather than guessing at a layout -- the netlist (.cir) stays authoritative.
KIND_SYMBOL: dict[str, str] = {
    "resistor": "res",
    "capacitor": "cap",
    "inductor": "ind",
    "voltage_source": "voltage",
    "current_source": "current",
    "diode": "diode",
    "zener": "diode",  # closest stock 2-pin symbol; BV lives in the .model line
    "led": "diode",
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


# Minimal ideal op-amp subcircuit (a high-gain VCVS). Portable to ngspice and
# LTspice; adequate for feedback amplifier topologies. Terminals: IN+ IN- OUT.
_OPAMP_SUBCKT = [
    ".subckt OPAMP inp inn out",
    "E1 out 0 inp inn 100k",
    ".ends",
]


def _ref(comp: Component) -> str:
    """Reference designator, prefixed to its SPICE element type if needed."""
    prefix = KIND_PREFIX[comp.kind]
    return comp.ref if comp.ref[:1].upper() == prefix else f"{prefix}{comp.ref}"


def to_netlist(circuit: Circuit) -> str:
    """Render ``circuit`` as SPICE netlist text."""
    gmap = _ground_map(circuit)
    lines: list[str] = [f"* {circuit.title}"]
    models: dict[str, str] = {}  # model name -> ".model ..." line (deduped)
    need_opamp = False

    for comp in circuit.components:
        ref = _ref(comp)
        nodes = [_node(n, gmap) for n in comp.nodes]

        if comp.kind == "diode":
            model = comp.model or "Dgeneric"
            models[model] = f".model {model} D"
            lines.append(f"{ref} {' '.join(nodes[:2])} {model}")

        elif comp.kind == "zener":
            model = comp.model or f"DZ_{ref}"
            models[model] = f".model {model} D(BV={comp.value or '5.1'})"
            lines.append(f"{ref} {' '.join(nodes[:2])} {model}")

        elif comp.kind == "led":
            # Generic red-LED parameters (~1.8-2V forward drop) -- good enough for
            # "does this light up / limit current correctly" teaching circuits.
            model = comp.model or "DLED"
            models[model] = f".model {model} D(IS=93.2p RS=42m N=3.73 EG=1.9 XTI=1)"
            lines.append(f"{ref} {' '.join(nodes[:2])} {model}")

        elif comp.kind in ("vcvs", "vccs"):
            # E/G: gain (V/V or A/V) senses the voltage between the two control nodes.
            lines.append(f"{ref} {' '.join(nodes[:4])} {comp.value or '1'}")

        elif comp.kind in ("cccs", "ccvs"):
            # F/H: gain (A/A or V/A) senses the current through a named V source.
            vref = comp.model or ""
            lines.append(f"{ref} {' '.join(nodes[:2])} {vref} {comp.value or '1'}".rstrip())

        elif comp.kind == "bjt":
            npn = (comp.subtype or "npn").lower() != "pnp"
            model = comp.model or ("QNPN" if npn else "QPNP")
            models[model] = f".model {model} {'NPN' if npn else 'PNP'}"
            # C B E; ngspice defaults give a working Gummel-Poon model.
            lines.append(f"{ref} {' '.join(nodes[:3])} {model}")

        elif comp.kind == "mosfet":
            nmos = (comp.subtype or "nmos").lower() not in ("pmos", "pfet", "p")
            model = comp.model or ("MNMOS" if nmos else "MPMOS")
            models[model] = f".model {model} {'NMOS' if nmos else 'PMOS'}"
            dgsb = nodes[:4]
            if len(dgsb) == 3:  # bulk defaults to source
                dgsb = dgsb + [dgsb[2]]
            lines.append(f"{ref} {' '.join(dgsb)} {model}")

        elif comp.kind == "opamp":
            need_opamp = True
            lines.append(f"{ref} {' '.join(nodes[:3])} OPAMP")

        else:  # two-terminal passives and sources
            lines.append(f"{ref} {' '.join(nodes[:2])} {comp.value}".rstrip())

    lines.extend(models.values())
    if need_opamp:
        lines.extend(_OPAMP_SUBCKT)

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
    Good enough to open, view, and hand-edit; not a tidy hand-drawn layout. Kinds
    with no stock 2-pin symbol (transistors, op-amps, dependent sources) are
    listed as a comment instead of placed -- the netlist (.cir) has them.
    """
    gmap = _ground_map(circuit)
    drawable = [c for c in circuit.components if c.kind in KIND_SYMBOL]
    skipped = [c for c in circuit.components if c.kind not in KIND_SYMBOL]

    spacing = 192
    x0, y0 = 96, 96
    width = max(880, x0 + len(drawable) * spacing + 200)

    lines = ["Version 4", f"SHEET 1 {width} 680"]
    for i, comp in enumerate(drawable):
        lines.extend(_asc_symbol_block(comp, x0 + i * spacing, y0, gmap))

    # SPICE analysis directive as a schematic directive (leading '!').
    directive = f".{circuit.analysis.type} {circuit.analysis.args}".rstrip()
    lines.append(f"TEXT {x0} {y0 + 320} Left 2 !{directive}")
    if skipped:
        # Plain schematic annotation (no leading '!' -- that marks a SPICE
        # directive, and this is just a note for whoever opens the file).
        refs = ", ".join(_ref(c) for c in skipped)
        lines.append(f"TEXT {x0} {y0 + 380} Left 2 Not drawn here (see .cir): {refs}")
    return "\n".join(lines) + "\n"
