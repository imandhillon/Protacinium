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
	#f = wave.open(r"./bangers/Damiano_Baldoni_-_Ive_not_fear.wav","rb")
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
	return np_music, data[0]
#borrowed from gruv. dont think it works as is. GRUV is on github for reference.
def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
	model = Sequential()
	#This layer converts frequency space to hidden space
	model.add(Dense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
	for cur_unit in xrange(num_recurrent_units):
		model.add(LSTM(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
	#This layer converts hidden space back to frequency space
	model.add(Dense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions))
	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	return model

music, rate = wav_to_np('./bangers/Damiano_Baldoni_-_Ive_not_fear.wav')
#flat_music = [item for sublist in music for item in sublist]
#wav.write('new.wav', rate, music)

#rate, music = wav.read('new.wav')
'''for x in music:
	print(x)'''

#play_music()

#Create data
X = []
Y = []
n_prev = 50000

for ind in xrange(len(music)-n_prev):
	x = music[ind:ind+n_prev]
	y = music[ind+n_prev]
	X.append(x)
	Y.append(y)

print(x)
seed = music[(randint(0,(len(music-n_prev)))):n_prev]

#this isnt working. breaks at 'fit' call
print('Building brain...')
model = Sequential()
model.add(LSTM(1024, input_shape=(n_prev, 2), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(1024, input_shape=(n_prev, 2), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('linear'))
print('Braining...')
optimizer = RMSprop(lr=0.01)
model.compile(loss='mse', optimizer='rmsprop')
model.fit(X, Y, batch_size=441000, epochs=5, verbose=2)
print('nsjs')
predict = []
x = seed
x = np.expand_dims(x, axis=0)
print(x)

for ind in xrange(441000):
	preds = model.predict(x)
	x = np.squeeze(x)
	x = np.concatenate((x, preds))
	x = x[1:]
	x = np.expand_dims(x, axis=0)
	preds = np.squeeze(preds)
	prediction.append(preds)