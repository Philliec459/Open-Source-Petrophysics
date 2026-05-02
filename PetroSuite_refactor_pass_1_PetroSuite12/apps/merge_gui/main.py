# apps/merge_gui/main.py
from __future__ import annotations

import os
import sys

from PySide6.QtWidgets import QApplication

from apps.merge_gui.ui_main_window import MainWindow


def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    w = MainWindow()
    w.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()