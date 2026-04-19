# merge_gui/ui/main_window.py  (same idea in petro_gui)

from PySide6.QtWidgets import QMainWindow, QDockWidget, QTabWidget
from PySide6.QtCore import Qt

from petrocore.viz.log_canvas_pg import LogCanvasPG
from petrocore.models.dataset import Dataset

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Merge/QC")
        self.resize(1600, 900)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Real log canvas widget
        self.log_canvas = LogCanvasPG()
        self.tabs.addTab(self.log_canvas, "Log Canvas")

        # Keep placeholders for now
        self.tabs.addTab(self._dock_placeholder("QC"), "QC")
        self.tabs.addTab(self._dock_placeholder("Alignment Score"), "Alignment Score")

        # Docks as before...
        self.addDockWidget(Qt.LeftDockWidgetArea, self._dock("Project", self._dock_placeholder("Project Tree")))
        self.addDockWidget(Qt.LeftDockWidgetArea, self._dock("Curve Inventory", self._dock_placeholder("Curve list + families")))
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock("Alignment", self._dock_placeholder("Bulk shift + anchors")))
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock("Merge", self._dock_placeholder("Merge params + preview")))
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock("Export", self._dock_placeholder("Export options")))

    def _dock(self, title, widget):
        d = QDockWidget(title, self)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        return d

    def _dock_placeholder(self, text):
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel(text))
        lay.addStretch(1)
        return w

    # Quick test hook (you can remove later)
    def demo_load(self, ds: Dataset):
        self.log_canvas.set_dataset(ds)
        self.log_canvas.set_tracks(self.log_canvas.standard_4track_template(ds))
