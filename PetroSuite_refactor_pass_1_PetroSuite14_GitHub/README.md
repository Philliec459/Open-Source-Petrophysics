# PetroSuite Qt Workflow

PetroSuite is a Qt-based petrophysical workflow app translated from notebook workflows. The app keeps Qt UI code under `apps/` and shared workflow/math code under `petrocore/`.

## Setup

From this project directory:

```bash
python -m pip install -r requirements.txt
```

## Launch

```bash
python -m apps.merge_gui.main
```

## Current Layout

- `apps/merge_gui/`: Qt application shell, panels, plugins, and widgets
- `petrocore/config/`: curve aliases, curve families, curve priorities, and curve-set configuration
- `petrocore/services/`: curve classification and winner-selection helpers
- `petrocore/workflow/`: reusable petrophysical calculations and plotting helpers
- `PetroSuite_Projects/`: saved sample projects and `.petro` well packages

## Basic QC Smoke Test

1. Launch the app.
2. Open a saved project from `PetroSuite_Projects/`.
3. Select and load a `.petro` well.
4. Confirm curve sets and tops populate.
5. Apply a zone of interest.
6. Run PHIT, Vsh, and Sw / Waxman-Smits workflows.
7. Confirm initial, workspace, and final depth plots render.
8. Save the well.
9. Reopen the saved well and confirm curves, tops, parameters, and zone reload.

## Notes

The central curve alias source is `petrocore/config/curve_aliases.py`. Prefer using the shared workflow functions in `petrocore/workflow/` for calculations rather than duplicating math in Qt panels.


## git commands

You do not need GitHub. You do not need branches at first. You do not need fancy commands. For now, only four ideas matter.

**1. A Repository**
A git repository is just a folder where git is watching changes.

For PetroSuite:

```bash
cd /Users/craig/Documents/Codex/PetroSuite_refactor_pass_1_PetroSuite12
git init
```

That turns the folder into a git repo.

**2. Status**
This asks, “What changed?”

```bash
git status
```

I use this constantly.

**3. Add**
This says, “Include these changed files in my next save point.”

```bash
git add .
```

The dot means “everything in this folder.”

**4. Commit**
This creates the save point.

```bash
git commit -m "Initial cleaned PetroSuite Qt workflow"
```

After that, you have a checkpoint.

The basic rhythm is:

```bash
git status
git add .
git commit -m "Short note about what changed"
```

Example:

```bash
git status
git add .
git commit -m "Add README and requirements"
```

Later, to see your save points:

```bash
git log --oneline
```

For your situation, git would let us say: “This is the cleaned-up working version. If the next refactor breaks anything, we can compare or recover.”

### These are the main commands.
cd /Users/craig/Documents/Codex/PetroSuite_refactor_pass_1_PetroSuite12
git init
git add .
git commit -m "Initial cleaned PetroSuite Qt workflow"