# apps/merge_gui/widgets/altair_crossplot_view.py

from pathlib import Path
import tempfile
import webbrowser

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton

from petrocore.workflow.altair_xplot import build_altair_dashboard


class AltairCrossplotView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_df = None
        self.last_html = None

        self.title_label = QLabel("Altair Xplot Dashboard")
        self.open_button = QPushButton("Open Altair Dashboard in Browser")
        self.open_button.clicked.connect(self.open_in_browser)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.open_button)

    def load_dashboard(self, df, df_chart=None, df_pef=None, df_pickett=None,
                       top=None, bottom=None):
        self.current_df = df

        chart = build_altair_dashboard(
            df=df,
            df_chart=df_chart,
            df_pef=df_pef,
            df_pickett=df_pickett,
            top=top,
            bottom=bottom,
        )

        tmp = Path(tempfile.gettempdir()) / "petrosuite_altair_dashboard.html"
        chart.save(str(tmp))
        self.last_html = tmp

        print(f"[AltairCrossplotView] wrote dashboard to: {tmp}")
        webbrowser.open(tmp.as_uri())

    def open_in_browser(self):
        if self.last_html and self.last_html.exists():
            webbrowser.open(self.last_html.as_uri())