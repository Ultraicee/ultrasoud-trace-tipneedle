import sys
import os
from PySide2.QtCore import QSize, Qt
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("myapp")

        button = QPushButton("press me")
        button.setCheckable(True)
        button.clicked.connect(self.the_button_was_clicked)
        self.setCentralWidget(button)

    def the_button_was_clicked(self):
        print("yes")


app = QApplication(sys.argv)
os.environ['QT_MAC_WANTS_LAYER'] = '1'
window = MainWindow()
window.show()
app.exec_()
