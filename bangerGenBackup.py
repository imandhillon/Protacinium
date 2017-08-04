from PyQt5 import QtWidgets, QtGui, QtCore
from os import listdir
from os.path import isfile, join
import sys
import recenttensor


class BangerWindow(QtWidgets.QMainWindow):
	
	def __init__(self):
		QtWidgets.QWidget.__init__(self)
		self.setWindowTitle('SoundGen')
		self.setStyleSheet("QMainWindow {border-image: url(music1.jpg) 0 0 0 0 stretch stretch; }")
		self.setFixedSize(400, 450)
		self.setup()
		
	def setup(self):
		self.frame = BangerFrame()
		self.setCentralWidget(self.frame)
		self.show()
		
		
class BangerFrame(QtWidgets.QWidget):
	def __init__(self):
		QtWidgets.QWidget.__init__(self)
		self.directoryText = ''
		self.onlyFiles = []
		self.setup()
		
	def setup(self):
		self.box = QtWidgets.QGridLayout()
		self.setLayout(self.box)
		
		self.title = QtWidgets.QLabel()
		self.title.setPixmap(QtGui.QPixmap('title.png'))
		
		self.play = PlayButton(QtGui.QPixmap('image.png'), self)
		self.play.clicked.connect(self.playSound)
		
		self.generate = QtWidgets.QPushButton("Generate Sounds", self)
		self.generate.clicked.connect(self.generateSound)
		
		self.directory = QtWidgets.QPushButton("Choose Directory")
		self.directory.clicked.connect(self.setDirectory)
		
		self.box.addWidget(self.title, 1 , 1, 1, 3)
		self.box.addWidget(self.directory, 3, 2, 1, 1)
		self.box.addWidget(self.generate, 4, 2, 1, 1)
		self.box.addWidget(self.play, 5, 2, 1, 1)
		
		self.show()
		
	def setDirectory(self):
		options = QtWidgets.QFileDialog.DontResolveSymlinks | QtWidgets.QFileDialog.ShowDirsOnly
		directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose Directory", self.directoryText, options=options)
		if directory:
			self.directoryText = directory	
		
	
	def playSound(self):
		try:
			recenttensor.play_music(self.filename)
		except:
			recenttensor.play_music('new.wav')
		
		#msg = "Did you like it?"
		#rater = QtWidgets.QMessageBox.question(self, 'Feedback', msg, QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
		
		#if rater == QtWidgets.QMessageBox.Yes:
			#print "coolio"
		#elif rater == QtWidgets.QMessageBox.No:
			#print "Maybe the next one then..."
		
	
	def generateSound(self):
		self.filename = recenttensor.run(self.directoryText)
		
	
		
class PlayButton(QtWidgets.QAbstractButton):
	def __init__(self, pixmap, parent = None):
		super(PlayButton, self).__init__(parent)
		self.pixmap = pixmap
		size = QtCore.QSize(20, 20)
		self.setIconSize(size)
		
	def paintEvent(self, event):
		painter = QtGui.QPainter(self)
		painter.drawPixmap(event.rect(), self.pixmap)
		
	def sizeHint(self):
		return self.pixmap.size()
	
			

if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	main_window = BangerWindow()
	app.exec_()