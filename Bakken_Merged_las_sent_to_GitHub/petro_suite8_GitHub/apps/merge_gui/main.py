from __future__ import annotations

import os
import sys
from PySide6.QtWidgets import QApplication

MERGED_LAS = "/Users/craig/Documents/petro_suite2/data/outputs/merged/Merged_Well_Log_Bakken_Bakken_renamed.las"




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


