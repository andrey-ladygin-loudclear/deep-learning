import sys

from PyQt5.QtCore import QRect, QMetaObject
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QPushButton, QLineEdit, QWidget, QMenuBar, QStatusBar


class Ui_MainWindow(object):

    def connect(self):
        self.updateWindow=QDialog()
        self.ui_update=Ui_Dialog()
        self.ui_update.setupUi(self.updateWindow)
        self.updateWindow.show()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(391, 248)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lineEdit = QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QRect(80, 60, 113, 27))
        self.lineEdit.setObjectName("lineEdit")
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QRect(80, 100, 112, 34))
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setGeometry(QRect(0, 0, 391, 31))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle("MainWindow")
        self.pushButton.setText("MainWindow")
        self.pushButton.clicked.connect(self.connect)

class Ui_Dialog(object):

    def save_data(self):
        d = self.lineEdit.text()
        print(d)

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(397, 219)
        self.pushButton = QPushButton(Dialog)
        self.pushButton.setGeometry(QRect(110, 100, 112, 34))
        self.pushButton.setObjectName("pushButton")
        self.lineEdit = QLineEdit(Dialog)
        self.lineEdit.setGeometry(QRect(110, 60, 113, 27))
        self.lineEdit.setText("")
        self.lineEdit.setObjectName("lineEdit")

        self.retranslateUi(Dialog)
        QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle("Dialog")
        self.pushButton.setText("Dialog")
        self.pushButton.clicked.connect(self.save_data)

class Dialog(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.close)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.onClicked)

    def onClicked(self):
        updateDialog = Dialog()
        updateDialog.exec_()
        self.lineEdit.setText(updateDialog.lineEdit.text())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())