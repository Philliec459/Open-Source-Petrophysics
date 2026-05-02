# PetroSuite Refactor Pass 1

Launch command remains:

```bash
python -m apps.merge_gui.main
```

## Main architectural change

This pass starts the paradigm shift:

- `apps/` keeps Qt widgets, buttons, combo boxes, messages, and launch wiring.
- `petrocore/` owns curve aliases, curve selection, and compute engines.

## New central curve source of truth

Added:

```text
petrocore/config/curve_aliases.py
```

This replaces scattered local lists such as:

```python
["TNPH", "NPHI", "CNL", "NPOR"]
["RHOB", "RHOZ", "DEN"]
["GR", "SGR", "CGR"]
```

Use these helpers everywhere:

```python
from petrocore.config.curve_aliases import best_curve, family_matches, family_candidates

best_curve(df.columns, "RHOB")
family_matches(df.columns, "TNPH")
family_candidates("GR")
```

## New pure petrocore compute engines

Added:

```text
petrocore/workflow/porosity_engine.py
petrocore/workflow/umaa_rhomaa.py
petrocore/workflow/runner.py
```

The plugin UI now calls petrocore instead of doing calculations directly in `apps`.

## Updated app files

Updated:

```text
apps/merge_gui/plugins/porosity_plugin.py
apps/merge_gui/plugins/umaa_rhomaa_plugin.py
apps/merge_gui/ui_panels/phit_panel.py
apps/merge_gui/ui_panels/vsh_panel.py
apps/merge_gui/ui_panels/sw_panel.py
apps/merge_gui/ui_panels/plots_panel_final.py
petrocore/viz/log_canvas_pg.py
petrocore/services/curve_family_service.py
```

## Important bug fixed

In `umaa_rhomaa_plugin.py`, the PHIT auto-pick was setting the wrong combo box in the original code. This is fixed.

## What this does NOT fully finish yet

This is not yet a full rewrite of every workflow. It is a safe first structural pass that keeps your current launch command and UI intact while centralizing the worst duplication. The next pass should move Vsh and Waxman-Smits calculations out of the Qt panels into `petrocore/workflow/vsh.py` and `petrocore/workflow/saturation/waxman_smits.py`.
