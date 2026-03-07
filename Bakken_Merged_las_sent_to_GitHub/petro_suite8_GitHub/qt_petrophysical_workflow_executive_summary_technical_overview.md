# Qt Petrophysical Workflow – Executive Summary & Technical Overview

## Executive Summary

This application implements a **Qt (PySide6)–based petrophysical workflow** that integrates multi‑run LAS data, parameterized saturation models, and interactive visualization into a single, reproducible system. The core design principle is simple but powerful:

> **All computations write into a single authoritative table (`state.analysis_df`), and all visualizations read from it.**

This guarantees consistency between calculations, plots, and exported parameters, while allowing iterative re‑computation as the user adjusts assumptions (Rw, m, n, Vsh model, etc.).

The workflow currently supports:

- Multi‑run LAS merge and curve family resolution
- Standard (Pickett/Archie‑style) saturation products
- Waxman–Smits saturation with physically consistent porosity partitioning
- Composite porosity/water visualization using PyQtGraph fill tracks
- Persistent parameter export (`Pickett.txt`) for reproducibility

---

## High‑Level Architecture

### Core Objects

- **`WorkflowController`**  
  Orchestrates computation and visualization refresh. Owns the active `state`.

- **`state`**  
  A lightweight container holding:
  - `analysis_df` – the master DataFrame of all curves (measured + derived)
  - `params` – the current parameter dictionary driven by the UI

- **UI Panels**
  - Compute panels (SW / WS tabs)
  - `PlotsPanel` (depth tracks and crossplots)

---

## Data Flow (Authoritative Pattern)

1. **Load / Merge LAS runs**  
   → populate `state.analysis_df`

2. **Resolve curve families**  
   (RT, PHIT, CBW, VSH, etc.) to actual mnemonics

3. **Compute products**  
   - Write *only* into `analysis_df`
   - Never compute directly inside plotting code

4. **Sync UI**  
   - `rebuild_view()` ensures widgets reference the latest DataFrame
   - `update_plots()` / `refresh_plots()` redraw tracks

This separation is what makes the system stable and extensible.

---

## Saturation Workflows

### 1) Standard SW (Pickett‑Style)

Triggered by:
```python
_on_compute_sw_clicked()
```

**Purpose**
- Compute Archie/Pickett‑style saturation and bulk volumes

**Typical outputs written to `analysis_df`:**
- `SWT`, `BVW`, `BVO`
- `MSTAR_FIT`, `MSTAR_APP`

These products are typically used for quick QC and chartbook‑style analysis.

---

### 2) Waxman–Smits (WS) Workflow

Triggered by:
```python
_on_compute_ws_clicked()
```

**Purpose**
- Compute Waxman–Smits saturation
- Partition porosity into physically meaningful components

**Key derived curves:**
- `SW_CP` – Waxman–Smits water saturation
- `BVWT_CP = PHIT * SW_CP`
- `BVWe_CP = max(BVWT_CP − CBW, 0)`
- `PHIE = max(PHIT − CBW, 0)`

**Important constraints**
- `PHIT` is clamped ≥ 0 before use
- `PHIE ≤ PHIT` enforced only after PHIT is non‑negative

---

## Qv and B (Physical Implementation)

### Qv (Hill–Shirley–Klein)

When `Swb` is available, Qv is computed as:

\[
Q_v = \frac{S_{wb}}{0.6425 / \sqrt{\rho_f \cdot SAL} + 0.22}
\]

- Requires `Swb`, `SAL`, and fluid density
- If `Swb` is missing, Qv falls back to zero (Archie‑like behavior)

### B (Bdacy)

- Computed from temperature and Rw using the **Bdacy correlation**
- Reduced to a **single scalar** (median) for Waxman–Smits

This ensures B is consistent and stable across the interval.

---

## Track‑Based Visualization

### Track Keys vs Curve Names (Critical Concept)

- **Track keys** (e.g. `"bvw"`) identify a plot area
- **Curve names** (e.g. `"PHIE"`) are data columns

Confusing the two leads to `KeyError` and empty tracks.

---

### Track 5 – Porosity & Water Partition (Composite Fill Track)

This track intentionally **does not use NMR**. It visualizes the Waxman–Smits porosity system:

| Region | Meaning |
|------|--------|
| PHIT → PHIE | Bound water porosity (CBW) |
| PHIE → BVWe | Effective bulk water |
| BVWe → 0 | Remaining effective pore volume |

Implemented using `FillBetweenItem` in PyQtGraph.

**Key rule:** composite tracks must *not* be indiscriminately cleared with `PlotItem.clear()` after plotting.

---

## UI Parameter Handling (Why Defaults Were Written)

Qt widgets do **not commit typed values** until focus changes.

### Fixes implemented

1. **Live synchronization**
   ```python
   spin.valueChanged.connect(lambda v: params[key] = v)
   ```

2. **Forced commit before compute/export**
   ```python
   spin.interpretText()
   ```

This guarantees `Pickett.txt` always reflects what the user sees.

---

## Parameter Export (Reproducibility)

At WS compute time, constants are written to:

```
./data/parameters/Pickett.txt
```

Contents include:
- `m_cem`
- `n_sat`
- `Rw`
- `mslope`
- `Bdacy`

This creates a durable record of assumptions used to generate results.

---

## Common Pitfalls and Lessons Learned

- **Empty tracks** → caused by clearing after plotting
- **Missing legends** → `addLegend()` must be called explicitly
- **Dark fills** → use RGBA brushes with low alpha
- **Dashed Rt** → set pen style (`Qt.DashLine`)
- **Qv = 0 everywhere** → `Swb` missing
- **Negative PHIE** → PHIT must be clamped before enforcing PHIE ≤ PHIT

---

## Design Philosophy (Why This Works)

- Single source of truth (`analysis_df`)
- Strict separation of compute vs display
- Physically interpretable saturation partitioning
- Parameter persistence for auditability

This architecture mirrors professional petrophysical platforms while remaining transparent, extensible, and reproducible.

---

## Recommended Next Enhancements

- YAML/JSON parameter export alongside `Pickett.txt`
- Optional Rw‑as‑curve Waxman–Smits variant
- Automated Swb derivation strategies (CBW‑based vs NMR‑based)
- Batch processing across wells with identical parameter sets

---

*End of document*

