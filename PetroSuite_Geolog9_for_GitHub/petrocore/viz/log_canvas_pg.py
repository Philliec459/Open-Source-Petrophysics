# packages/petrocore/petrocore/viz/log_canvas_pg.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg

from petrocore.models.dataset import Dataset


@dataclass
class TrackSpec:
    name: str
    curves: List[str] = field(default_factory=list)  # curve mnemonics to plot
    x_range: Optional[tuple[float, float]] = None    # optional fixed x-range
    log_x: bool = False                               # resistivity track etc.
    show_grid: bool = True


class LogCanvasPG(QWidget):
    """
    A Geolog-style log canvas starter:
      - multiple tracks (columns)
      - shared depth axis (y)
      - per-track curve lists
      - supports log-x track for resistivity
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._dataset: Optional[Dataset] = None
        self._tracks: List[TrackSpec] = []

        pg.setConfigOptions(antialias=True)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.glw)

        self._plot_items: List[pg.PlotItem] = []
        self._curves_items: Dict[tuple[int, str], pg.PlotDataItem] = {}
        
        # Optional: override pen per curve name (e.g., base vs moving)
        # self.curve_pens: Dict[str, pg.QtGui.QPen] = {}
        self.curve_pens: Dict[str, pg.QtGui.QPen] = {}

    def set_dataset(self, ds: Dataset):
        self._dataset = ds
        self.refresh()

    def set_tracks(self, tracks: List[TrackSpec]):
        self._tracks = tracks
        self.refresh()

    def refresh(self):
        self.glw.clear()
        self._plot_items = []
        self._curves_items = {}

        if self._dataset is None or self._dataset.data.empty or not self._tracks:
            # placeholder
            p = self.glw.addPlot()
            p.hideAxis("left")
            p.hideAxis("bottom")
            p.addItem(pg.TextItem("Load data + define tracks", color="k"))
            return

        ds = self._dataset
        depth = ds.depth.values.astype(float)

        # Create track plots side-by-side
        first_plot: Optional[pg.PlotItem] = None

        for i, tr in enumerate(self._tracks):
            p = self.glw.addPlot(row=0, col=i)
            self._plot_items.append(p)

            # Depth axis
            p.invertY(True)                      # depth increases downward
            p.showAxis("left", i == 0)           # show y-axis only on first track
            p.showAxis("bottom", True)

            if tr.show_grid:
                p.showGrid(x=True, y=True, alpha=0.2)

            p.setLabel("bottom", tr.name, color="k")
            p.setLabel("left", "DEPT", units="ft", color="k")

            # Link Y (depth) between tracks
            if first_plot is None:
                first_plot = p
            else:
                p.setYLink(first_plot)

            # Optional log-x (e.g., resistivity)
            if tr.log_x:
                p.setLogMode(x=True, y=False)

            # Plot each curve in this track
            for curve in tr.curves:
                if curve not in ds.data.columns:
                    continue
                x = pd.to_numeric(ds.data[curve], errors="coerce").values.astype(float)
                m = np.isfinite(x) & np.isfinite(depth)
                if m.sum() < 2:
                    continue

                # Create line (default pen; you can later map family->style)
 
                pen = self.curve_pens.get(curve, pg.mkPen(width=1.2))

                
                item = p.plot(x[m], depth[m], pen=pen, name=curve)
                
                self._curves_items[(i, curve)] = item

            # Set x range if provided
            if tr.x_range is not None:
                p.setXRange(tr.x_range[0], tr.x_range[1], padding=0.02)

        # Nice: add a shared legend on first track
        if self._plot_items:
            leg = self._plot_items[0].addLegend(offset=(10, 10))
            # pyqtgraph legend picks up curve names from plot(name=...)

    # Convenience helpers for your “templates”
    @staticmethod
    def standard_4track_template(ds: Dataset) -> List[TrackSpec]:
        # Use your family preference lists if present
        gr  = ds.best_curve_for_family("GR")  or ds.first_present(["HSGR", "GR", "SGR"])
        rhb = ds.best_curve_for_family("RHOB") or ds.first_present(["RHOZ", "RHOB"])
        tnp = ds.best_curve_for_family("TNPH") or ds.first_present(["TNPH", "NPHI", "NPOR"])
        rt  = ds.best_curve_for_family("RT")   or ds.first_present(["AT90", "AF90", "ILD", "RT"])

        tracks = [
            TrackSpec(name="GR", curves=[c for c in [gr] if c], x_range=(0, 150)),
            TrackSpec(name="Porosity", curves=[c for c in [rhb, tnp] if c]),
            TrackSpec(name="Resistivity", curves=[c for c in [rt] if c], log_x=True),
            TrackSpec(name="Computed", curves=[]),
        ]
        return tracks
