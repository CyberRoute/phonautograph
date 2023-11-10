import sys
from phonautograph import AudioRecorderPlayer
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioRecorderPlayer()
    window.show()
    sys.exit(app.exec_())
