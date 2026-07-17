"""Render a ``Circuit`` as a real schematic (schemdraw) for visual verification.

There is no netlist importer in schemdraw, so this lays the circuit out itself:
the source sits on the left, series components run left-to-right along a top rail,
shunt components drop to a bottom ground rail, and ground hangs off the rail. This
covers the ladder-style topologies most hand-drawn sketches use (dividers, RC/RL/RLC
filters, rectifiers); unusual topologies still draw, just less tidily.

``render_schematic`` returns PNG bytes for ``st.image``.
"""

from __future__ import annotations

import re

from sketch2spice.model import KIND_TERMINALS, Circuit, Component

DX = 3.0       # horizontal spacing between top-rail nodes
Y_TOP = 3.0    # top rail height
Y_GND = 0.0    # ground rail height
SHUNT_DX = 1.4  # offset for a 2nd/3rd shunt on the same node

# Kinds with more than two terminals -- these don't fit the ladder layout, so a
# circuit containing any of them is drawn with the net-label renderer instead.
_MULTI = {"bjt", "mosfet", "opamp"}
_TWO_TERM = {"resistor", "capacitor", "inductor", "voltage_source", "current_source", "diode"}


def _is_gnd(net: str, ground: str) -> bool:
    return net == ground or net == "0"


def _element(comp: Component):
    import schemdraw.elements as elm

    k = comp.kind
    sub = (comp.subtype or "").lower()
    if k == "resistor":
        return elm.Resistor()
    if k == "capacitor":
        return elm.Capacitor()
    if k == "inductor":
        return elm.Inductor()
    if k == "diode":
        return elm.Diode()
    if k == "current_source":
        return elm.SourceI()
    if k == "voltage_source":
        if comp.value and re.search(r"sin|pulse|ac\b", comp.value, re.I):
            return elm.SourceSin()
        return elm.SourceV()
    if k == "bjt":
        return elm.BjtPnp() if sub == "pnp" else elm.BjtNpn()
    if k == "mosfet":
        return elm.PFet() if sub in ("pmos", "pfet", "p") else elm.NFet()
    if k == "opamp":
        return elm.Opamp()
    return elm.Resistor()


def _terminal_anchors(comp: Component) -> list[tuple[str, str]]:
    """(schemdraw anchor, human terminal label) per terminal, in nodes order."""
    k = comp.kind
    if k == "bjt":
        return [("collector", "C"), ("base", "B"), ("emitter", "E")]
    if k == "mosfet":
        return [("drain", "D"), ("gate", "G"), ("source", "S")]
    if k == "opamp":  # nodes order is IN+, IN-, OUT
        return [("in2", "IN+"), ("in1", "IN-"), ("out", "OUT")]
    t = KIND_TERMINALS.get(k, ["n1", "n2"])
    return [("start", t[0]), ("end", t[1])]


def _label(comp: Component) -> str:
    detail = comp.value or (comp.model or "")
    return f"{comp.ref}\n{detail}" if detail else comp.ref


def _label_color(comp: Component) -> str:
    # Low-confidence reads get a red label so they stand out against the photo.
    return "red" if comp.confidence < 0.6 else "black"


def _order_nets(circuit: Circuit, ground: str, src: Component | None) -> list[str]:
    """Left-to-right order of the non-ground (top-rail) nets."""
    comps = circuit.components

    def series(c):
        return c is not src and not _is_gnd(c.nodes[0], ground) and not _is_gnd(c.nodes[1], ground)

    # Start from the source's high side, else any non-ground net.
    if src:
        a, b = src.nodes[0], src.nodes[1]
        high = b if _is_gnd(a, ground) and not _is_gnd(b, ground) else a
    else:
        high = next((n for c in comps for n in c.nodes if not _is_gnd(n, ground)), None)

    order: list[str] = []
    if high is not None:
        order.append(high)
        used: set[int] = set()
        advanced = True
        while advanced:
            advanced = False
            last = order[-1]
            for i, c in enumerate(comps):
                if i in used or not series(c):
                    continue
                n0, n1 = c.nodes[0], c.nodes[1]
                nxt = n1 if (n0 == last and n1 not in order) else (n0 if (n1 == last and n0 not in order) else None)
                if nxt is not None:
                    order.append(nxt)
                    used.add(i)
                    advanced = True
                    break

    # Any remaining non-ground nets (parallel branches) go to the right.
    for c in comps:
        for n in c.nodes:
            if n and not _is_gnd(n, ground) and n not in order:
                order.append(n)
    return order


def _render_net_labels(circuit: Circuit) -> bytes:
    """Schematic with net-name labels on terminals (for active-device circuits).

    The ladder layout only handles two-terminal parts, so once a transistor or
    op-amp is present we draw each component with its real symbol on a grid and
    tag every terminal with its net name. It reads like a schematic with net
    labels (a legitimate style) and works for any topology.
    """
    import schemdraw

    schemdraw.use("matplotlib")
    comps = [c for c in circuit.components if c.nodes]

    d = schemdraw.Drawing()
    cols, cw, ch = 3, 5.5, 5.0
    for i, comp in enumerate(comps):
        row, col = divmod(i, cols)
        e = _element(comp).at((col * cw, -row * ch))
        if comp.kind in _TWO_TERM:
            e = e.right()
        e = e.label(_label(comp), loc="top", color=_label_color(comp))
        for (anchor, term), net in zip(_terminal_anchors(comp), comp.nodes):
            e = e.label(f"{term}={net}", loc=anchor, fontsize=9, color="#2f6fb0")
        d += e

    d.draw(show=False)
    return d.get_imagedata("png")


def render_schematic(circuit: Circuit) -> bytes:
    """Draw ``circuit`` as a schematic and return PNG bytes."""
    import schemdraw

    if any(c.kind in _MULTI for c in circuit.components):
        return _render_net_labels(circuit)

    schemdraw.use("matplotlib")

    ground = circuit.ground_node or "0"
    comps = [c for c in circuit.components if len(c.nodes) >= 2 and c.nodes[0] and c.nodes[1]]
    sources = [c for c in comps if c.kind in ("voltage_source", "current_source")]
    src = sources[0] if sources else None

    order = _order_nets(circuit, ground, src)
    x = {net: i * DX for i, net in enumerate(order)}

    def series(c):
        return c is not src and not _is_gnd(c.nodes[0], ground) and not _is_gnd(c.nodes[1], ground)

    def shunt(c):
        return c is not src and (_is_gnd(c.nodes[0], ground) != _is_gnd(c.nodes[1], ground))

    d = schemdraw.Drawing()
    rail_xs: list[float] = []       # x positions that touch the ground rail
    degree: dict[str, int] = {}     # connections per top net (for junction dots)

    def bump(net):
        if not _is_gnd(net, ground):
            degree[net] = degree.get(net, 0) + 1

    # Source on the left, drawn from the ground rail *up* to the top rail.
    # (schemdraw's source symbol mis-sizes when drawn downward, so go up.)
    if src and (src.nodes[0] in x or src.nodes[1] in x):
        top_net = src.nodes[0] if src.nodes[0] in x else src.nodes[1]
        xs = x[top_net]
        d += _element(src).at((xs, Y_GND)).to((xs, Y_TOP)).label(_label(src), loc="left", color=_label_color(src))
        rail_xs.append(xs)
        bump(top_net)

    # Series components along the top rail.
    for c in comps:
        if not series(c):
            continue
        a, b = c.nodes[0], c.nodes[1]
        if a not in x or b not in x:
            continue
        (xa, xb) = sorted((x[a], x[b]))
        d += _element(c).at((xa, Y_TOP)).to((xb, Y_TOP)).label(_label(c), color=_label_color(c))
        bump(a)
        bump(b)

    # Shunt components dropping to the ground rail; offset duplicates on a node.
    shunt_count: dict[str, int] = {}
    for c in comps:
        if not shunt(c):
            continue
        top_net = c.nodes[1] if _is_gnd(c.nodes[0], ground) else c.nodes[0]
        if top_net not in x:
            continue
        k = shunt_count.get(top_net, 0)
        shunt_count[top_net] = k + 1
        xs = x[top_net] + k * SHUNT_DX
        if k:  # stub across the top rail to the offset column
            d += schemdraw.elements.Line().at((x[top_net], Y_TOP)).to((xs, Y_TOP))
        d += _element(c).at((xs, Y_TOP)).to((xs, Y_GND)).label(_label(c), color=_label_color(c))
        rail_xs.append(xs)
        bump(top_net)

    # Ground rail + symbol.
    if rail_xs:
        lo, hi = min(rail_xs), max(rail_xs)
        if hi > lo:
            d += schemdraw.elements.Line().at((lo, Y_GND)).to((hi, Y_GND))
        d += schemdraw.elements.Ground().at(((lo + hi) / 2, Y_GND))

    # Junction dots where three or more connections meet a node.
    for net, deg in degree.items():
        if deg >= 3:
            d += schemdraw.elements.Dot().at((x[net], Y_TOP))

    d.draw(show=False)
    return d.get_imagedata("png")
