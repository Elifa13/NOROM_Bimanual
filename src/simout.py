"""
simout.py - Simulink SimulationOutput (.mat, MCOS) reader without MATLAB.

The .mat files in Data/Raw are MATLAB v5 files whose single variable `out`
is a Simulink.SimulationOutput object. scipy cannot decode MCOS objects
directly, but the underlying data lives in the file's function workspace
(the MCOS "FileWrapper__" cell array). This module digs it out.
"""
from __future__ import annotations

import io
import numpy as np
import scipy.io as sio
from scipy.io.matlab._mio5 import MatFile5Reader
from scipy.io.matlab._mio5_params import mat_struct


# ---------------------------------------------------------------- low level

def _read_filewrapper(path):
    """Return the MCOS FileWrapper__ cell array (n,1) of object storage."""
    top = sio.loadmat(path, struct_as_record=False)
    raw = top["__function_workspace__"].tobytes()
    # rebuild a valid 128-byte MAT header in front of the workspace stream
    stream = io.BytesIO(b" " * 124 + raw[:4] + raw[8:])
    r = MatFile5Reader(stream, struct_as_record=False)
    r.initialize_read()
    r.mat_stream.seek(128)
    hdr, _ = r.read_var_header()
    v = r.read_var_array(hdr)
    return v[0, 0].MCOS[0]["arr"]


def _unwrap(x):
    """Peel 1x1 object wrappers off until we hit real content."""
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.ravel()[0]
    return x


def _txt(x):
    x = _unwrap(x)
    if isinstance(x, np.ndarray) and x.dtype.kind in "US":
        r = x.ravel()
        return str(r[0]) if r.size else ""
    if isinstance(x, str):
        return x
    return ""


# --------------------------------------------------------------- high level

class Trial:
    """One .mat file = one trial (one Simulink run)."""

    def __init__(self, path):
        self.path = str(path)
        cells = _read_filewrapper(path)
        self._cells = cells
        self.vars = None      # mat_struct: the SimulationOutput variable store
        self.meta = {}        # ModelName, timestamps, solver info, ...
        for i in range(cells.shape[0]):
            e = _unwrap(cells[i, 0])
            if isinstance(e, mat_struct):
                f = set(e._fieldnames)
                if "tout" in f:
                    self.vars = e
                elif "ModelName" in f:
                    for k in e._fieldnames:
                        self.meta[k] = _unwrap(getattr(e, k))
                elif "WallClockTimestampStart" in f:
                    for k in ("WallClockTimestampStart", "WallClockTimestampStop",
                              "TotalElapsedWallTime"):
                        if k in f:
                            self.meta[k] = _unwrap(getattr(e, k))
        if self.vars is None:
            raise ValueError(f"no SimulationOutput variable store found in {path}")

    # -- names -------------------------------------------------------------
    @property
    def names(self):
        return list(self.vars._fieldnames)

    def describe(self):
        """List every logged variable with shape / source block / labels."""
        rows = []
        for f in self.names:
            v = _unwrap(getattr(self.vars, f))
            if isinstance(v, mat_struct) and "signals" in v._fieldnames:
                t = _unwrap(v.time)
                sa = getattr(v, "signals")
                if not isinstance(sa, np.ndarray):
                    sa = np.array([[sa]], dtype=object)
                for k in range(sa.size):
                    ss = _unwrap(sa.ravel()[k])
                    vals = _unwrap(ss.values)
                    rows.append(dict(
                        name=f, port=k,
                        label=_txt(getattr(ss, "label", "")),
                        block=_txt(v.blockName),
                        n=None if t is None else np.shape(t)[0],
                        shape=getattr(vals, "shape", None),
                        kind="struct-with-time",
                    ))
            elif isinstance(v, np.ndarray) and v.dtype.kind == "f":
                rows.append(dict(name=f, port=0, label="", block="",
                                 n=v.shape[0], shape=v.shape, kind="array"))
            else:
                rows.append(dict(name=f, port=0, label="", block="",
                                 n=None, shape=getattr(v, "shape", None),
                                 kind="other/unresolved"))
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows

    # -- access ------------------------------------------------------------
    def get(self, name, port=0):
        """Return signal values as a float array (squeezed)."""
        v = _unwrap(getattr(self.vars, name))
        if isinstance(v, mat_struct) and "signals" in v._fieldnames:
            sa = getattr(v, "signals")
            if not isinstance(sa, np.ndarray):
                sa = np.array([[sa]], dtype=object)
            ss = _unwrap(sa.ravel()[port])
            return np.asarray(_unwrap(ss.values), float).squeeze()
        return np.asarray(v, float).squeeze()

    def time(self, name=None):
        if name is None:
            return np.asarray(_unwrap(self.vars.tout), float).squeeze()
        v = _unwrap(getattr(self.vars, name))
        return np.asarray(_unwrap(v.time), float).squeeze()

    def to_frame(self, names=None):
        """Wide DataFrame of every 1-D signal sampled on the main clock."""
        import pandas as pd
        t = self.time()
        n = t.size
        cols = {"t": t}
        for f in (names or self.names):
            if f == "tout":
                continue
            v = _unwrap(getattr(self.vars, f))
            if isinstance(v, mat_struct) and "signals" in v._fieldnames:
                sa = getattr(v, "signals")
                if not isinstance(sa, np.ndarray):
                    sa = np.array([[sa]], dtype=object)
                for k in range(sa.size):
                    ss = _unwrap(sa.ravel()[k])
                    a = np.asarray(_unwrap(ss.values), float)
                    if a.ndim == 2 and a.shape[0] == n:
                        for c in range(a.shape[1]):
                            key = f if (sa.size == 1 and a.shape[1] == 1) else f"{f}_{k}{c}"
                            cols[key] = a[:, c]
            elif isinstance(v, np.ndarray) and v.dtype.kind == "f" and v.shape[0] == n:
                cols[f] = v.squeeze()
        return pd.DataFrame(cols)


def load(path):
    return Trial(path)
