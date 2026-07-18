"""Run a SPICE netlist and read back the waveforms.

Simulation runs on **ngspice** (headless, reliable, scriptable): the Wine-based
LTspice 26 for macOS does not run batch simulations dependably from the CLI. The
LTspice ``.cir``/``.asc`` files we generate are still for opening in the LTspice GUI;
the automatic plots come from ngspice. Netlists are largely portable, so this is a
runner swap only -- ``sketch2spice.netlist.to_ngspice`` handles the small dialect gaps.

``spicelib``'s ``RawRead`` parses the resulting ``.raw`` (its format is the shared
Berkeley/ngspice raw format).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sketch2spice.netlist import to_ngspice

NGSPICE = shutil.which("ngspice") or "ngspice"


@dataclass
class SimResult:
    x_name: str  # "time", "frequency", ...
    x: np.ndarray
    traces: dict[str, np.ndarray] = field(default_factory=dict)
    complex_traces: dict[str, np.ndarray] = field(default_factory=dict)
    log: str = ""

    def signal_names(self) -> list[str]:
        return list(self.traces.keys())

    @property
    def is_ac(self) -> bool:
        """True for an AC sweep, where traces carry magnitude *and* phase."""
        return self.x_name.lower().startswith("freq") and bool(self.complex_traces)


def run(netlist_text: str, work_dir: str | Path | None = None) -> SimResult:
    """Simulate ``netlist_text`` with ngspice and return the resulting traces.

    Raises ``RuntimeError`` (with the ngspice log attached) if the run fails.
    """
    from spicelib import RawRead

    work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="sketch2spice_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    net_path = work_dir / "circuit.net"
    raw_path = work_dir / "circuit.raw"
    net_path.write_text(to_ngspice(netlist_text))

    proc = subprocess.run(
        [NGSPICE, "-b", "-r", str(raw_path), str(net_path)],
        capture_output=True,
        text=True,
        cwd=str(work_dir),
        timeout=120,
    )
    log = f"$ {NGSPICE} -b -r circuit.raw circuit.net\n\n{proc.stdout}\n{proc.stderr}"

    if not raw_path.exists():
        raise RuntimeError(f"ngspice produced no .raw output.\n\n--- log ---\n{log}")

    raw = RawRead(str(raw_path))
    names = raw.get_trace_names()
    if not names:
        raise RuntimeError(f"ngspice .raw had no traces.\n\n--- log ---\n{log}")

    # ngspice tags its raw axis differently than spicelib's get_axis() expects, so
    # read the first trace (time / frequency) directly as the x axis.
    x_name = names[0]
    axis = np.asarray(raw.get_trace(x_name).get_wave(), dtype=complex)
    axis = np.abs(axis) if np.any(axis.imag) else axis.real

    traces: dict[str, np.ndarray] = {}
    complex_traces: dict[str, np.ndarray] = {}
    for name in names[1:]:
        wave = np.asarray(raw.get_trace(name).get_wave(), dtype=complex)
        # Real analyses give real waves; magnitude+phase for genuinely complex (AC)
        # results -- keep the complex wave too so callers can plot phase / a Bode.
        if np.any(wave.imag):
            traces[name] = np.abs(wave)
            complex_traces[name] = wave
        else:
            traces[name] = wave.real

    return SimResult(x_name=x_name, x=axis, traces=traces, complex_traces=complex_traces, log=log)
