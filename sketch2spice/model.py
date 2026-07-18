"""Shared intermediate representation for a parsed circuit.

The vision backends produce a ``Circuit``; the netlist generator consumes one.
Keeping a single Pydantic model in the middle means the review table, the JSON
schema handed to the vision models, and the SPICE emitter all agree on one shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Components we support. Two-terminal passives/sources plus multi-terminal active
# devices (BJT, MOSFET, op-amp). Adding a kind means: a prefix in KIND_PREFIX,
# terminal names in KIND_TERMINALS, a branch in netlist.to_netlist, and a symbol
# in viz. -- and a mention in the vision prompt.
ComponentKind = Literal[
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

AnalysisKind = Literal["tran", "ac", "op", "dc"]

# SPICE reference-designator prefix for each kind. The first letter of a SPICE
# element line determines how the simulator interprets it. Op-amps are emitted as
# a subcircuit instance ("X").
KIND_PREFIX: dict[str, str] = {
    "resistor": "R",
    "capacitor": "C",
    "inductor": "L",
    "voltage_source": "V",
    "current_source": "I",
    "diode": "D",
    "zener": "D",
    "led": "D",
    "bjt": "Q",
    "mosfet": "M",
    "opamp": "X",
    "vcvs": "E",
    "vccs": "G",
    "cccs": "F",
    "ccvs": "H",
}

# Terminal names per kind, in the order they appear in Component.nodes. Used for
# the review-table legend and to know how many terminals each device has. The
# MOSFET bulk (4th) is optional -- it defaults to the source when omitted.
KIND_TERMINALS: dict[str, list[str]] = {
    "resistor": ["n1", "n2"],
    "capacitor": ["n1", "n2"],
    "inductor": ["n1", "n2"],
    "voltage_source": ["+", "-"],
    "current_source": ["+", "-"],
    "diode": ["anode", "cathode"],
    "zener": ["anode", "cathode"],
    "led": ["anode", "cathode"],
    "bjt": ["C", "B", "E"],
    "mosfet": ["D", "G", "S", "B"],
    "opamp": ["IN+", "IN-", "OUT"],
    # Dependent sources: E (VCVS) and G (VCCS) sense a voltage across two control
    # terminals; F (CCCS) and H (CCVS) instead sense current through a named
    # controlling voltage source, given in Component.model rather than a node.
    "vcvs": ["OUT+", "OUT-", "CTRL+", "CTRL-"],
    "vccs": ["OUT+", "OUT-", "CTRL+", "CTRL-"],
    "cccs": ["OUT+", "OUT-"],
    "ccvs": ["OUT+", "OUT-"],
}


class Component(BaseModel):
    ref: str = Field(description="Reference designator, e.g. R1, C2, V1.")
    kind: ComponentKind = Field(description="What the component is.")
    nodes: list[str] = Field(
        description=(
            "Net names this component connects, in order. Two-terminal parts have "
            "exactly two. Use '0' for ground. For a voltage/current source the first "
            "node is the + terminal."
        ),
    )
    value: str = Field(
        default="",
        description=(
            "Component value as written on the sketch, e.g. '1k', '10u', '5', or a "
            "source spec like 'SINE(0 5 1k)' or 'DC 5'. Empty if not legible."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "Optional SPICE .model name (for diodes, transistors, MOSFETs). For a "
            "cccs/ccvs (current-controlled source), this is instead the ref of the "
            "voltage source whose current it senses, e.g. 'V1'."
        ),
    )
    subtype: str | None = Field(
        default=None,
        description=(
            "Device polarity/variant when it matters: 'npn'/'pnp' for a BJT, "
            "'nmos'/'pmos' for a MOSFET. Ignored for other kinds."
        ),
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How confident the reading of this component is, 0-1.",
    )


class Analysis(BaseModel):
    type: AnalysisKind = Field(
        default="tran",
        description="Which SPICE analysis to run.",
    )
    args: str = Field(
        default="5m",
        description=(
            "Arguments for the analysis directive, e.g. '5m' for .tran, "
            "'dec 100 1 1Meg' for .ac. Not including the leading dot or type."
        ),
    )


class Circuit(BaseModel):
    title: str = Field(default="Circuit from sketch", description="Human-readable title.")
    ground_node: str = Field(
        default="0",
        description="The net treated as ground. SPICE requires ground to be node '0'.",
    )
    components: list[Component] = Field(default_factory=list)
    analysis: Analysis = Field(default_factory=Analysis)
    notes: str = Field(
        default="",
        description=(
            "Anything ambiguous or illegible in the sketch that a human should "
            "double-check before simulating."
        ),
    )


def circuit_json_schema() -> dict:
    """JSON schema for ``Circuit``, for the vision models' structured output."""
    return Circuit.model_json_schema()
