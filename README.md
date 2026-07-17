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
./run.sh                                    # syncs deps and starts the app
./run.sh --vision-url http://HOST:PORT/v1   # point at your own model endpoint
./run.sh --server.port 9000                 # extra args pass through to streamlit
```

The vision endpoint defaults to `http://127.0.0.1:8080/v1`. You can change it in the
app's sidebar (**Local model endpoint**), or set the default with `--vision-url` /
the `LOCAL_VISION_URL` env var (include the trailing `/v1` — it's an OpenAI-compatible
base URL). Or run directly: `uv run streamlit run app.py`.

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

Supported parts: resistors, capacitors, inductors, voltage/current sources, diodes,
BJTs, MOSFETs, and op-amps, with transient / AC / operating-point / DC analyses.
Transistors use ngspice's default device models; op-amps are an ideal high-gain VCVS
subcircuit (good for feedback amplifiers). Set a component's `subtype` (npn/pnp,
nmos/pmos) in the review table for transistors. Adding another device kind means: a
prefix in `model.py` (`KIND_PREFIX`/`KIND_TERMINALS`), a branch in `netlist.py`, a
symbol in `viz.py`, and a mention in the vision prompt.
