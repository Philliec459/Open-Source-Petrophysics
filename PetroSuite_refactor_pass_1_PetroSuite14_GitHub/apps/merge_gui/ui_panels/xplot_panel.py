# xplot_panel.py


from PySide6.QtWidgets import QWidget, QVBoxLayout, QListWidget, QLabel


class XPlotPanel(QWidget):
    def __init__(self, main_window=None, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self._build_ui()

    def _state(self):
        try:
            return self.main_window.controller.get_state()
        except AttributeError:
            print("[XPlotPanel] controller/state not available")
            return None


    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Crossplot Launchers")
        title.setStyleSheet("font-size: 11pt; font-weight: bold;")
        layout.addWidget(title)

        self.launch_list = QListWidget()
        self.launch_list.addItems([
            "Neutron-Density",
            "Sonic-Neutron",
            "Sonic-Density",
            "PEF-Bulk Density",
            "UMAA-RHOMAA",
            "POTA-THOR",
            "POTA-PEF",
            "Pickett Plot",
            "Buckles Plot",
            "Steiber Plot",
            "Generic XY",
            "Generic Regression XY",
            "Histogram",
            "Altair Xplot",
        ])

        self.launch_list.itemClicked.connect(self._on_launch_clicked)

        layout.addWidget(self.launch_list)

    def _on_launch_clicked(self, item):
        name = item.text()
        print(f"[XPlotPanel] clicked: {name}")

        if self.main_window is None:
            print("[XPlotPanel] main_window not set")
            return

        self.main_window.launch_xplot(name)