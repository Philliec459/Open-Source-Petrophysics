
# apps/merge_gui/main.py



import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


print("cwd:", os.getcwd())
print("sys.path[0:3]:", sys.path[:3])



import numpy as np
import pandas as pd
from PySide6.QtWidgets import QApplication

from ui_main_window import MainWindow          # local import (same folder)
from petrocore.models.dataset import Dataset   # shared backend model


def main():
    app = QApplication(sys.argv)
    w = MainWindow()

    # -----------------------------
    # Demo dataset (remove later)
    # -----------------------------
    depth = np.arange(8000, 9000, 0.5)
    df = pd.DataFrame(
        {
            "GR":   60 + 20*np.sin(depth/50),
            "RHOZ": 2.45 - 0.1*np.sin(depth/80),
            "TNPH": 0.18 + 0.05*np.cos(depth/60),
            "RT":   10**(1 + 0.4*np.sin(depth/100)),
        },
        index=depth,
    )

    ds = Dataset(
        data=df,
        families_map={
            "GR":   ["HSGR", "GR", "SGR"],
            "RHOB": ["RHOZ", "RHOB"],
            "TNPH": ["TNPH", "NPHI", "NPOR"],
            "RT":   ["AT90", "AF90", "ILD", "RT"],
        },
    )

    # If your MainWindow has demo_load(), this will populate the log canvas
    if hasattr(w, "demo_load"):
        w.demo_load(ds)

    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
