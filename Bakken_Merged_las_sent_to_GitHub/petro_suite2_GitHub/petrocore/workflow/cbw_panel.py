]# petrocore/workflows/cbw_panel.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import panel as pn
import pandas as pd


def cbw_intercept_picker(
    analysis_df: pd.DataFrame,
    *,
    vsh_col: str = "VSH_HL",
    cbw_col: str = "CBW",
    start: float = 0.0,
    end: float = 0.5,
    step: float = 0.01,
    value: float = 0.15,
):
    slider = pn.widgets.FloatSlider(name="CBW_Intercept", start=start, end=end, step=step, value=value)

    def _plot(CBW_Int: float):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_title("Vsh_HL vs. CBW", color="blue")

        ax.plot(
            pd.to_numeric(analysis_df[vsh_col], errors="coerce").to_numpy(),
            pd.to_numeric(analysis_df[cbw_col], errors="coerce").to_numpy(),
            "r.",
        )

        x = np.linspace(0.0, 1.0, 200)
        ax.plot(x, x * CBW_Int, "k-")

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 0.5)
        ax.set_ylabel("CBW [v/v]", color="blue")
        ax.set_xlabel("Vsh_HL [v/v]", color="blue")
        ax.grid(True)
        fig.tight_layout()
        plt.close(fig)
        return fig

    ui = pn.Column(pn.interact(_plot, CBW_Int=slider), slider)
    return ui, slider
