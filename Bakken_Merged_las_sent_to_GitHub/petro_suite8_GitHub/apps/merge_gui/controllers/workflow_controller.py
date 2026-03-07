# apps/merge_gui/controllers/workflow_controller.py

from __future__ import annotations
from typing import Optional
import pandas as pd

from petrocore.models.dataset import Dataset

class WorkflowController:
    def __init__(self, state, plots_panel, console):
        self.state = state
        self.plots_panel = plots_panel
        self.console = console

    # -------------------------
    # Dataset
    # -------------------------
    def set_dataset(self, dataset: Dataset):
        self.state.dataset = dataset
    
        dmin, dmax = float(dataset.depth.min()), float(dataset.depth.max())
        self.state.depth_limits = (dmin, dmax)
    
        self.state.analysis_df = dataset.data
        self.state.analysis_df_view = dataset.data
    
        self.console.append(
            f"Dataset loaded: {dataset.name} | curves={len(dataset.curves())} | depth={dmin:.1f}-{dmax:.1f}"
        )
    
        self.refresh_plots()
    

    def set_depth_range(self, top: float, base: float):
        """
        Called by DepthRangePanel or TopsPanel.
        Sets analysis window from full dataset.
        """
        try:
            t = float(top)
            b = float(base)
        except Exception:
            return
    
        if b < t:
            t, b = b, t
    
        self.state.depth_top = t
        self.state.depth_base = b
    
        ds = getattr(self.state, "dataset", None)
        if ds is None:
            return
    
        df = ds.data
    
        try:
            win = df.loc[t:b].copy()
        except Exception:
            win = df
    
        self.state.analysis_df = win
        self.state.analysis_df_view = win
    
        self.console.append(f"Analysis window set: {t:.2f} – {b:.2f} ft | rows={len(win)}")
    
        self.refresh_plots()


    def refresh_plots(self):
        ds = getattr(self.state, "dataset", None)
        if ds is None:
            return
    
        # full vs zoi
        plot_ds = ds.zoi if (getattr(ds, "zoi_depth_range", None) is not None) else ds
    
        # Always keep the full df available
        self.state.analysis_df = plot_ds.data
    
        # If a depth range is set, build a view; otherwise clear it
        t = getattr(self.state, "depth_top", None)
        b = getattr(self.state, "depth_base", None)
    
        if t is not None and b is not None:
            try:
                t = float(t); b = float(b)
                if b < t:
                    t, b = b, t
                self.state.analysis_df_view = self.state.analysis_df.loc[t:b].copy()
            except Exception:
                self.state.analysis_df_view = self.state.analysis_df
        else:
            self.state.analysis_df_view = None
    
        self.plots_panel.update_all(self.state)
    

    # -------------------------
    # ZOI
    # -------------------------
    def set_zoi(self, top: float, base: float):
        ds: Optional[Dataset] = getattr(self.state, "dataset", None)
        if ds is None:
            self.console.append("⚠️ No dataset loaded.")
            return

        z = ds.set_zoi(top, base)
        self.console.append(f"ZOI set: {ds.zoi_depth_range[0]:.1f}-{ds.zoi_depth_range[1]:.1f} ft | rows={len(z.data)}")
        self.refresh_plots()

    def clear_zoi(self):
        ds: Optional[Dataset] = getattr(self.state, "dataset", None)
        if ds is None:
            return
        ds.clear_zoi()
        self.console.append("ZOI cleared.")
        self.refresh_plots()


    def update_plots(self):
        """Back-compat for UI panels that call update_plots()."""
        self.refresh_plots()