"""Shared intermediate representation for a parsed circuit.

The vision backends produce a ``Circuit``; the netlist generator consumes one.
Keeping a single Pydantic model in the middle means the review table, the JSON
schema handed to the vision models, and the SPICE emitter all agree on one shape.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Two-terminal parts we support today. Adding a transistor or op-amp means adding
# a kind here and a branch in netlist.to_netlist -- nothing else changes.
ComponentKind = Literal[
    "resistor",
    "capacitor",
    "inductor",
    "voltage_source",
    "current_source",
    "diode",
]

AnalysisKind = Literal["tran", "ac", "op", "dc"]

# SPICE reference-designator prefix for each kind. The first letter of a SPICE
# element line determines how the simulator interprets it.
KIND_PREFIX: dict[str, str] = {
    "resistor": "R",
    "capacitor": "C",
    "inductor": "L",
    "voltage_source": "V",
    "current_source": "I",
    "diode": "D",
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
        description="Optional SPICE .model name (mainly for diodes).",
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
