from __future__ import annotations

import pandas as pd

from petrocore.workflow.phit_chartbook import compute_phit_chartbook
from petrocore.workflow.phit_rms import compute_phit_rms


class MainController:
    def __init__(self, state, main_window=None):
        self.state = state
        self.main_window = main_window

        if not hasattr(self.state, "params") or not isinstance(self.state.params, dict):
            self.state.params = {}

        if not hasattr(self.state, "view_df"):
            self.state.view_df = None

        if not hasattr(self.state, "zoi_top"):
            self.state.zoi_top = None

        if not hasattr(self.state, "zoi_base"):
            self.state.zoi_base = None

    # ---------------------------------------------------------
    # Basic access
    # ---------------------------------------------------------
    def get_state(self):
        return self.state

    # ---------------------------------------------------------
    # Parameter helpers
    # ---------------------------------------------------------
    def get_param(self, key, default=None):
        params = getattr(self.state, "params", None)
        if not isinstance(params, dict):
            self.state.params = {}
            params = self.state.params
        return params.get(key, default)

    def set_param(self, key, value):
        params = getattr(self.state, "params", None)
        if not isinstance(params, dict):
            self.state.params = {}
            params = self.state.params
        params[key] = value

    def update_params(self, mapping: dict):
        params = getattr(self.state, "params", None)
        if not isinstance(params, dict):
            self.state.params = {}
            params = self.state.params
        params.update(mapping)

    # ---------------------------------------------------------
    # Depth / ZoI helpers
    # ---------------------------------------------------------
    def _find_depth_col(self, df):
        if df is None or df.empty:
            return None

        for col in df.columns:
            cu = str(col).strip().upper()
            if cu in ["DEPT", "DEPTH", "MD"]:
                return col

        return None

    def _get_zoi_range(self):
        zoi_top = getattr(self.state, "zoi_top", None)
        zoi_base = getattr(self.state, "zoi_base", None)

        # backward compatibility
        if zoi_top is None:
            zoi_top = getattr(self.state, "depth_top", None)
        if zoi_base is None:
            zoi_base = getattr(self.state, "depth_base", None)

        if zoi_top is None or zoi_base is None:
            return None, None

        top = min(zoi_top, zoi_base)
        base = max(zoi_top, zoi_base)
        return top, base

    def _get_view_mask(self, df):
        if df is None or df.empty:
            return None

        depth_col = self._find_depth_col(df)
        if depth_col is None:
            return pd.Series(True, index=df.index)

        top, base = self._get_zoi_range()
        if top is None or base is None:
            return pd.Series(True, index=df.index)

        depth = pd.to_numeric(df[depth_col], errors="coerce")
        mask = (depth >= top) & (depth <= base)
        return mask.fillna(False)

    def rebuild_view(self):
        """
        Build state.view_df from state.analysis_df using the current ZoI.
        """
        df = getattr(self.state, "analysis_df", None)

        if df is None or df.empty:
            self.state.view_df = None
            return

        mask = self._get_view_mask(df)
        if mask is None:
            self.state.view_df = None
            return

        self.state.view_df = df.loc[mask].copy().reset_index(drop=True)

    def set_zoi(self, top_depth, base_depth):
        self.state.zoi_top = float(top_depth)
        self.state.zoi_base = float(base_depth)
        self.rebuild_view()
        self.refresh_ui()
        self.refresh_plots()

    def clear_zoi(self):
        self.state.zoi_top = None
        self.state.zoi_base = None
        self.rebuild_view()
        self.refresh_ui()
        self.refresh_plots()

    # ---------------------------------------------------------
    # Refresh helpers
    # ---------------------------------------------------------
    def refresh_plots(self):
        self.rebuild_view()

        if self.main_window is not None:
            self.main_window.refresh_plots()
        else:
            print("[WARN] MainController has no main_window")

    def refresh_ui(self):
        self.rebuild_view()

        if self.main_window is not None:
            self.main_window.refresh_all_panels()
        else:
            print("[WARN] MainController has no main_window")

    # ---------------------------------------------------------
    # PHIT calculation
    # ---------------------------------------------------------

    def calculate_phit(self, rhob_curve, nphi_curve, method="Density-Neutron Chartbook", chart=None):

        df = getattr(self.state, "analysis_df", None)

        if df is None or df.empty:
            print("[PHIT] No data loaded")
            return

        try:
            if method != "Density-Neutron Chartbook":
                print(f"[PHIT] Unsupported method: {method}")
                return

            work_df = df.copy()
            mask = self._get_view_mask(work_df)

            if mask is None or not mask.any():
                print("[PHIT] No rows available in selected zone")
                return

            subset = work_df.loc[mask].copy()

            out_subset = compute_phit_chartbook(
                subset,
                rhob_curve=rhob_curve,
                nphi_curve=nphi_curve,
                chartbook_key=chart
            )


            for col in out_subset.columns:
                if col not in work_df.columns:
                    work_df[col] = pd.NA

            new_cols = [c for c in out_subset.columns if c not in subset.columns or c in ["POR_DEN", "PHIT"]]
            for col in new_cols:
                if col in out_subset.columns:
                    work_df.loc[mask, col] = out_subset[col].values

            self.state.analysis_df = work_df

            self.set_param("phit.method", method)
            self.set_param("phit.rhob_curve", rhob_curve)
            self.set_param("phit.nphi_curve", nphi_curve)
            self.set_param("phit.chart", chart)

            top, base = self._get_zoi_range()
            if top is not None and base is not None:
                print(f"[PHIT] Done using {rhob_curve} + {nphi_curve} within ZoI {top:.2f} to {base:.2f}")
            else:
                print(f"[PHIT] Done using {rhob_curve} + {nphi_curve} over full well")

            self.refresh_ui()
            self.refresh_plots()

        except Exception as e:
            print(f"[PHIT] Failed: {e}")



    def calculate_phit_rms(self, rhob_curve, nphi_curve, method="Density-Neutron RMS"):
        df = getattr(self.state, "analysis_df", None)

        if df is None or df.empty:
            print("[PHIT] No data loaded")
            return

        try:
            if method != "Density-Neutron RMS":
                print(f"[PHIT] Unsupported method: {method}")
                return

            work_df = df.copy()
            mask = self._get_view_mask(work_df)

            if mask is None or not mask.any():
                print("[PHIT] No rows available in selected zone")
                return

            subset = work_df.loc[mask].copy()

            out_subset = compute_phit_rms(
                subset,
                rhob_curve=rhob_curve,
                nphi_curve=nphi_curve,
                matrix_density=2.71,
                fluid_density=1.1,
            )


            for col in out_subset.columns:
                if col not in work_df.columns:
                    work_df[col] = pd.NA

            new_cols = [c for c in out_subset.columns if c not in subset.columns or c in ["POR_DEN", "PHIT"]]
            for col in new_cols:
                if col in out_subset.columns:
                    work_df.loc[mask, col] = out_subset[col].values

            self.state.analysis_df = work_df

            self.set_param("phit.method", method)
            self.set_param("phit.rhob_curve", rhob_curve)
            self.set_param("phit.nphi_curve", nphi_curve)

            top, base = self._get_zoi_range()
            if top is not None and base is not None:
                print(f"[PHIT] Done using {rhob_curve} + {nphi_curve} within ZoI {top:.2f} to {base:.2f}")
            else:
                print(f"[PHIT] Done using {rhob_curve} + {nphi_curve} over full well")

            self.refresh_ui()
            self.refresh_plots()

        except Exception as e:
            print(f"[PHIT] Failed: {e}")



