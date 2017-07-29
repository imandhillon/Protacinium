from PyQt5 import QtWidgets, QtGui, QtCore
from os import listdir
from os.path import isfile, join
import sys
import fancyschmancystaticgenerator


class BangerWindow(QtWidgets.QMainWindow):
	
	def __init__(self):
		QtWidgets.QWidget.__init__(self)
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
		self.grid = QtWidgets.QGridLayout()
		self.setLayout(self.grid)
		
		self.title = QtWidgets.QLabel("Welcome!")
		self.title.setAlignment(QtCore.Qt.AlignCenter)
		
		self.play = QtWidgets.QPushButton("Play Sounds", self)
		self.play.clicked.connect(self.playSound)
		
		self.generate = QtWidgets.QPushButton("Generate Sounds", self)
		self.generate.clicked.connect(self.generateSound)
		
		self.directory = QtWidgets.QPushButton("Choose Directory")
		self.directory.clicked.connect(self.setDirectory)

		self.grid.addWidget(self.title, 1, 1, 1, 3)
		self.grid.addWidget(self.directory, 2, 2, 1, 1)
		self.grid.addWidget(self.generate, 3, 2, 1, 1)
		self.grid.addWidget(self.play, 5, 2, 1, 1)
		
		self.show()
		
	def setDirectory(self):
		options = QtWidgets.QFileDialog.DontResolveSymlinks | QtWidgets.QFileDialog.ShowDirsOnly
		directory = QtWidgets.QFileDialog.getExistingDirectory(self, "QFileDialog.getDirectory()", self.directoryText, options=options)
		if directory:
			self.onlyFiles = [f for f in listdir(directory) if isfile(join(directory, f))]
			self.directoryText = directory	
		
	def playSound(self):
		learn.play_music()
		
	def generateSound(self):
		x = self.onlyFiles[0]
		print "TEST: " + x
		
		for x in self.onlyFiles:
			path = self.directoryText+ '/' + x
			#print path
			learn.run(path)


if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	main_window = BangerWindow()
	app.exec_()