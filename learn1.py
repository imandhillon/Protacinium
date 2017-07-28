from __future__ import print_function
from random import randint

import os
import numpy as np
import scipy.io.wavfile as wav
import wave
import pyaudio
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.optimizers import RMSprop

def play_music():
	#define stream chunk
	chunk = 1024
	#open a wav file
	#f = wave.open(r"./notstatic/Damiano_Baldoni_-_Ive_not_fear.wav","rb")
	f = wave.open(r"new.wav","rb")
	#instantiate PyAudio
	p = pyaudio.PyAudio()
	#open stream
	stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
					channels = f.getnchannels(),
					rate = f.getframerate(),
					output = True)
	
	data = f.readframes(chunk)
	print(str(data))

	#play stream
	while data: 
		stream.write(data)
		data = f.readframes(chunk)

	#stop stream
	stream.stop_stream()
	stream.close()

	#close PyAudio  
	p.terminate()

def rand_wav():
	data = np.random.uniform(-1,1,44100) # 44100 random samples between -1 and 1
	scaled = np.int16(data/np.max(np.abs(data)) * 32767)
	wav.write('test.wav', 44100, scaled)

def convert_folder_to_wav(directory, sample_rate=44100):
	for file in os.listdir(directory):
		fullfilename = directory+file
		if file.endswith('.mp3'):
			convert_mp3_to_wav(filename=fullfilename, sample_frequency=sample_rate)
	return directory + 'wave/'

def wav_to_np(filename):
	data = wav.read(filename)
	np_music = data[1].astype('float32') / 32767.0
	print(np_music)
	return np_music, data[0]

def np_to_sample(music, block_size):
	blocks = []
	total_samples = music.shape[0]
	num_samples = 0
	while(num_samples < total_samples):
		block = music[num_samples:num_samples+block_size]
		if(block.shape[0] < block_size):
			padding = np.zeros((block_size - block.shape[0], 2))
			block = np.concatenate((block, padding))
		blocks.append(block)
		num_samples += block_size
	return blocks

def load_corpus(file, block_size):
	y_np = np.empty
	music, rate = wav_to_np(file)
	x_t = np_to_sample(music, block_size)
	#is list of numpy arrays of numpy arrays of floats
	#print((type(x_t)), type(x_t[1]), type(x_t[1][1]), type(x_t[1][1][1]))
	y_t = x_t[1:]					#If x, then y. Sorta
	y_t.append(np.zeros(block_size)) 	#make it fit

	y_left = y_t[:][:][:1]
	y_right = y_t[:][:][1:2]
	y_np = np.asarray(y_left)
	y_right = np.asarray(y_right)
	y_np = np.column_stack((y_np, y_right))

	x_left = x_t[:][:][:1]
	x_right = x_t[:][:][1:2]
	x_np = np.asarray(x_left)
	x_right = np.asarray(x_right)
	x_np = np.column_stack((x_np, x_right))
	#print(y_np)
	return x_np, y_np

music, rate = wav_to_np('./notstatic/Damiano_Baldoni_-_Ive_not_fear.wav')

if __name__ == '__main__':
	#wav.write('new.wav', rate, music)
	chunks_X = []
	chunks_Y = []
	X, Y_t = load_corpus('./notstatic/Damiano_Baldoni_-_Ive_not_fear.wav', block_size=2048)
	print(type(Y_t), type(Y_t[0]), type(Y_t[0][0]), type(Y_t[0][0][0]))

	#Not sure if necessary yet
	'''cur_seq = 0
	total_seq = len(X)
	print total_seq
	print max_seq_len
	while cur_seq + max_seq_len < total_seq:
		chunks_X.append(X[cur_seq:cur_seq+max_seq_len])
		chunks_Y.append(Y[cur_seq:cur_seq+max_seq_len])
		cur_seq +x = seed #is empty...
	= max_seq_len'''

	#seed = X[:][738:748]
	#tries to do brain stuff
	print('Building brain...')
	model = Sequential()
	model.add(LSTM(2048, input_shape=(4096, 2), return_sequences=True))
	model.add(Dropout(0.2))
	#model.add(LSTM(2048, input_shape=(4096, 2), return_sequences=False))
	#model.add(Dropout(0.2))
	model.add(Dense(2))
	model.add(Activation('linear'))

	print('Braining...')
	optimizer = RMSprop(lr=0.01)
	model.compile(loss='mse', optimizer='rmsprop')
	model.fit(X, Y_t, batch_size=441000, epochs=1, verbose=2)

	print('Doing brain stuff...')
	prediction = []
	#x = seed #is empty...
	print(X)
	print('#######\n')

	for ind in xrange(40):
		preds = model.predict(X)
		print(preds)
		print(len(preds))

		#copy-pasta'd from somewhere. Does not seem to help in my case
		x = np.squeeze(x)
		x = np.concatenate((x, preds))
		x = x[1:]
		x = np.expand_dims(x, axis=0)
		preds = np.squeeze(preds)
		prediction.append(preds)
