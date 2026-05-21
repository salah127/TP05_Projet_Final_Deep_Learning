"""
extract_weights.py — Export best_model.keras weights to weights.json

Requires ONLY h5py (auto-installed if missing, ~4 MB).
NO TensorFlow needed — reads the HDF5 directly from the .keras ZIP.

  python3.13 extract_weights.py

After running, serve with:
    python3.13 -m http.server 8000
Then open: http://localhost:8000/demo.html
"""

import sys, os, json, zipfile, io, subprocess

# ── Check dependencies ────────────────────────────────────────────────
try:
    import h5py
    import numpy as np
except ImportError:
    print("[ERROR] h5py is not installed.")
    print(f"  Run:  {sys.executable} -m pip install h5py")
    print("  Then re-run this script.")
    sys.exit(1)

MODEL = "best_model.keras"
if not os.path.exists(MODEL):
    print(f"[ERROR] {MODEL} not found. Run this script from the project folder.")
    sys.exit(1)

# ── Extract the .h5 weights file from inside the .keras ZIP ──────────
print(f"Reading {MODEL}  ({os.path.getsize(MODEL) / 1024:.1f} KB)…")
with zipfile.ZipFile(MODEL) as z:
    contents = z.namelist()
    print(f"  Archive contents: {contents}")
    h5_name = next((n for n in contents if n.endswith(".h5")), None)
    if not h5_name:
        print("[ERROR] No .h5 weight file found inside the archive.")
        sys.exit(1)
    h5_bytes = z.read(h5_name)

# ── Walk the HDF5 and collect every numeric dataset ───────────────────
arrays = []

def _collect(node):
    if isinstance(node, h5py.Dataset):
        arr = np.array(node)
        if arr.ndim >= 1:          # skip scalars
            arrays.append(arr)
    elif hasattr(node, "keys"):
        for k in node.keys():
            _collect(node[k])

with h5py.File(io.BytesIO(h5_bytes), "r") as h:
    _collect(h)

print(f"  Found {len(arrays)} arrays with shapes: {[a.shape for a in arrays]}")

# ── Identify layers by shape (unique for our architecture) ────────────
# Model: Input(13) → Dense(64,relu) → Dropout(0.2) → Dense(32,relu) → Dense(3,softmax)
# Kernels: (13,64)  (64,32)  (32,3)
# Biases:  (64,)    (32,)    (3,)
by_shape = {}
for a in arrays:
    if a.shape not in by_shape:
        by_shape[a.shape] = a

EXPECTED = [
    ("dense",   (13, 64), (64,)),
    ("dense_1", (64, 32), (32,)),
    ("dense_2", (32,  3),  (3,)),
]

missing = [name for name, ks, bs in EXPECTED if ks not in by_shape or bs not in by_shape]
if missing:
    print(f"[ERROR] Could not locate weights for layers: {missing}")
    print("  Available shapes:", list(by_shape.keys()))
    sys.exit(1)

# ── Build output dict ─────────────────────────────────────────────────
out = {}
for name, kernel_shape, bias_shape in EXPECTED:
    W = by_shape[kernel_shape]
    b = by_shape[bias_shape]
    out[name] = {"W": W.tolist(), "b": b.tolist()}
    print(f"  {name}: W={W.shape}, b={b.shape}")

with open("weights.json", "w") as f:
    json.dump(out, f, separators=(",", ":"))

kb = os.path.getsize("weights.json") / 1024
print(f"\n[OK] weights.json  ({kb:.1f} KB)  — {len(out)} layers")
print("\nNext steps:")
print("  python3.13 -m http.server 8000")
print("  open http://localhost:8000/demo.html")

