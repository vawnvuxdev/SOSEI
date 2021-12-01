import sys
from app_ui import Ui_MainWindow

from PyQt5.QtWidgets import QApplication, QMainWindow

class AppWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.setupUi(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    appWindow = AppWindow()
    appWindow.show()
    sys.exit(app.exec())
