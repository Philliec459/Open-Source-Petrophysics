from __future__ import annotations

import os
import sys
from PySide6.QtWidgets import QApplication






def main():
    import sys
    from PySide6.QtWidgets import QApplication
    from apps.merge_gui.ui_main_window import MainWindow

    app = QApplication(sys.argv)

    w = MainWindow()
    w.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()


