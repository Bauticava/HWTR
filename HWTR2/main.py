from gui2 import QApplication, Window
import sys

#Main file to run the GUI
def main():
    app = QApplication(sys.argv)
    ex = Window()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
