# sketch2spice

Photograph a hand-drawn circuit sketch and turn it into a runnable simulation.

A vision model reads the photo into a structured netlist, you review and correct it,
and the circuit is simulated with the waveforms plotted — all in a local web app.
LTspice `.cir` netlist and `.asc` schematic files are produced for opening in the
LTspice GUI.

## Pipeline

```
photo → vision model → editable review table → SPICE netlist → simulate → plot
                                                      └→ download .cir / .asc
```

## Requirements

- Python ≥ 3.11 and [`uv`](https://docs.astral.sh/uv/).
- **ngspice** for simulation: `brew install ngspice`.
- A vision backend (pick one):
  - **Local (default):** an OpenAI-compatible server with a multimodal model at
    `http://127.0.0.1:8080` (e.g. a `llama.cpp` server). No API key.
  - **Claude:** set `ANTHROPIC_API_KEY` and select "Claude API" in the sidebar.
- LTspice (optional) only to open the generated `.cir`/`.asc` in its GUI.

## Setup

```sh
uv venv
uv pip install -e .
```

## Run

```sh
uv run streamlit run app.py
```

Then: upload a photo → **Analyze sketch** → check the redrawn schematic against your
photo and correct anything the model misread in the table (low-confidence reads are
labelled red) → **Regenerate netlist from table** → **Run LTspice** to simulate and
plot. Download the `.cir` / `.asc` from the buttons.

## Why ngspice, not LTspice, for the actual run

LTspice for macOS (v26+) is a Wine-wrapped Windows build whose headless batch CLI does
not run simulations reliably. `ngspice` is a native, scriptable, headless SPICE engine,
and SPICE netlists are portable, so simulation runs on ngspice while the LTspice-format
files remain available for the GUI. `sketch2spice/netlist.py:to_ngspice` bridges the two
dialects (`SINE(` → `SIN(`, and giving `.tran` the two arguments ngspice requires).

## Configuration (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `VISION_BACKEND` | `local` | `local` or `anthropic` |
| `LOCAL_VISION_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible base URL |
| `LOCAL_VISION_MODEL` | `serve-rpc` | model id on the local server |
| `ANTHROPIC_MODEL` | `claude-opus-4-8` | Claude model for the `anthropic` backend |

## Layout

| File | Role |
|---|---|
| `sketch2spice/model.py` | `Circuit` / `Component` intermediate representation |
| `sketch2spice/vision.py` | pluggable vision backends (local llama.cpp, Claude) |
| `sketch2spice/netlist.py` | `Circuit` → SPICE `.cir` / LTspice `.asc`; ngspice dialect fix |
| `sketch2spice/viz.py` | `Circuit` → schematic drawing (schemdraw) for visual review |
| `sketch2spice/simulate.py` | run ngspice, parse `.raw` into traces |
| `app.py` | Streamlit UI |

## Scope

Two-terminal parts today: resistors, capacitors, inductors, voltage/current sources,
and diodes, with transient / AC / operating-point / DC analyses. Transistors and op-amps
are the intended next step — add a `kind` in `model.py` and a branch in `netlist.py`.
