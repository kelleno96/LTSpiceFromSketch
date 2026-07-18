"""Electrical rule check: catch wiring/value mistakes before they hit ngspice.

A bad parse (or a bad edit in the review table) otherwise fails deep inside
ngspice with a cryptic singular-matrix error. This is a pure, deterministic
pass over the ``Circuit`` IR that catches the common causes early and explains
them in circuit terms -- advisory only, it never blocks simulation.
"""

from __future__ import annotations

import re
from typing import Literal, NamedTuple

from sketch2spice.model import KIND_TERMINALS, Circuit
from sketch2spice.netlist import parse_si

Severity = Literal["error", "warning"]

_VALUE_KINDS = {"resistor", "capacitor", "inductor", "voltage_source", "current_source"}
_NUMERIC_KINDS = {"resistor", "capacitor", "inductor"}
_SOURCE_KINDS = {"voltage_source", "current_source"}
_TIME_VARYING = re.compile(r"\b(SIN|SINE|PULSE|PWL|EXP|SFFM)\b", re.IGNORECASE)


class Finding(NamedTuple):
    severity: Severity
    message: str
    ref: str | None = None


def check(circuit: Circuit) -> list[Finding]:
    """Run all rule checks and return findings (empty if the circuit looks clean)."""
    if not circuit.components:
        return []

    findings: list[Finding] = []
    ground = circuit.ground_node or "0"
    seen_refs: set[str] = set()
    net_terminal_count: dict[str, int] = {}
    has_source = False
    has_ac_source = False
    has_time_varying_source = False

    for c in circuit.components:
        if c.ref in seen_refs:
            findings.append(Finding("error", f"duplicate reference designator '{c.ref}'", c.ref))
        seen_refs.add(c.ref)

        min_terminals = 3 if c.kind == "mosfet" else len(KIND_TERMINALS.get(c.kind, ["n1", "n2"]))
        if len(c.nodes) < min_terminals:
            terms = "/".join(KIND_TERMINALS.get(c.kind, []))
            findings.append(
                Finding(
                    "error",
                    f"{c.ref} ({c.kind}) needs {min_terminals} terminal(s) ({terms}) "
                    f"but only has {len(c.nodes)}",
                    c.ref,
                )
            )

        if c.kind in _VALUE_KINDS and not c.value.strip():
            findings.append(Finding("error", f"{c.ref} has no value set", c.ref))
        elif c.kind in _NUMERIC_KINDS and c.value.strip() and parse_si(c.value) is None:
            findings.append(
                Finding(
                    "warning",
                    f"{c.ref} value '{c.value}' doesn't look like a number "
                    "(e.g. '1k', '4.7u', '10meg')",
                    c.ref,
                )
            )

        if len(c.nodes) >= 2 and c.kind not in ("bjt", "mosfet", "opamp"):
            if c.nodes[0] and c.nodes[0] == c.nodes[1]:
                findings.append(
                    Finding("warning", f"{c.ref} has both terminals on net '{c.nodes[0]}' (shorted)", c.ref)
                )

        for n in c.nodes:
            if n:
                net_terminal_count[n] = net_terminal_count.get(n, 0) + 1

        if c.kind in _SOURCE_KINDS:
            has_source = True
            val = c.value or ""
            if re.search(r"\bAC\b", val, re.IGNORECASE):
                has_ac_source = True
            if _TIME_VARYING.search(val):
                has_time_varying_source = True

    if ground not in net_terminal_count:
        findings.append(Finding("error", f"no component connects to the ground net '{ground}'"))

    for net, count in net_terminal_count.items():
        if net != ground and count == 1:
            findings.append(Finding("warning", f"net '{net}' only has one connection (floating)"))

    if circuit.analysis.type == "ac" and has_source and not has_ac_source:
        findings.append(
            Finding(
                "warning",
                "AC analysis is selected but no source has an AC magnitude "
                "(e.g. value 'AC 1') -- the sweep will show no signal",
            )
        )
    if circuit.analysis.type == "tran" and has_source and not has_time_varying_source:
        findings.append(
            Finding(
                "warning",
                "transient analysis with only DC sources -- expect flat waveforms; "
                "did you mean a time-varying source like SINE(...)?",
            )
        )

    return findings
