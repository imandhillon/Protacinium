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

def write_np_as_wav(X, sample_rate, filename):
	Xnew = X * 32767.0
	Xnew = Xnew.astype('int16')
	wav.write(filename, sample_rate, Xnew)
	return

def convert_sample_blocks_to_np_audio(blocks):
	song_np = np.concatenate(blocks)
	return song_np

def rand_wav():
	data = np.random.uniform(-1,1,44100) # 44100 random samples between -1 and 1
	scaled = np.int16(data/np.max(np.abs(data)) * 32767)
	wav.write('test.wav', 44100, scaled)

def wav_dir_to_np(directory, sample_rate=44100):
	for file in os.listdir(directory):
		fullfilename = directory+file
		
	return directory + 'wave/'

def wav_to_np(filename):
	data = wav.read(filename)
	np_music = data[1].astype('float32') / 32767.0
	return np_music, data[0]

def np_to_sample(music, block_size=2048):
	blocks = []
	total_samples = music.shape[0]
	num_samples = 0
	while(num_samples < total_samples):
		block = music[num_samples:num_samples+block_size]
		if(block.shape[0] < block_size):
			padding = np.zeros((block_size - block.shape[0]))
			block = np.concatenate((block, padding))
		blocks.append(block)
		num_samples += block_size
	return blocks

'''def load_corpus(file, block_size=2048):
	#Have it handle directories *********
	music, rate = wav_to_np(file)
	#Convert stereo to mono ***Check sound quality after this step***
	music = music.sum(axis=1)/2

	#print(music, len(music))
	x_t = np_to_sample(music, block_size)

	#print((type(x_t)), type(x_t[1]), type(x_t[1][1]), type(x_t[1][1][1]))
	y_t = x_t[1:]					#If x, then y. Sorta
	y_t.append(np.zeros(block_size)) 	#make it fit
	serialize_corpus(x_t, y_t)'''

def serialize_corpus(x_train, y_train, seq_len=215):
	seqs_x = []
	seqs_y = []
	cur_seq = 0
	total_seq = len(x_train)
	print('total seq: ', total_seq)
	print('max seq: ', seq_len)

	while cur_seq + seq_len < total_seq:
		seqs_x.append(x_train[cur_seq:cur_seq+max_seq_len])
		seqs_y.append(y_train[cur_seq:cur_seq+max_seq_len])
		cur_seq += seq_len

	return seqs_x, seqs_y

def make_tensors(seqs_x, seqs_y, nb_examples, block_size):
	'''Have it handle directories *********'''
	music, rate = wav_to_np(file)
	#Convert stereo to mono ***Check sound quality after this step***
	music = music.sum(axis=1)/2

	#print(music, len(music))
	x_t = np_to_sample(music, block_size)

	#print((type(x_t)), type(x_t[1]), type(x_t[1][1]), type(x_t[1][1][1]))
	y_t = x_t[1:]					#If x, then y. Sorta
	y_t.append(np.zeros(block_size)) 	#make it fit
	seqs_x, seqs_y = serialize_corpus(x_t, y_t)

	'''Pretty much taken from GRUV. Updated it a little. It seemed to be deprecated.'''
	nb_examples = len(seqs_x)
	#nb_output_dims = block_size * 2
	#output_shape = (nb_examples, max_seq_len, block_size)
	x_data = np.zeros(output_shape).astype('float32')
	y_data = np.zeros(output_shape).astype('float32')

	print('\nCalculating mean and variance and saving data\n')
	x_data = np.array(seqs_x)
	y_data = np.array(seqs_y)
	#Runs with half of examples to save time and memory
	for examples in xrange(nb_examples/2):
		for seqs in xrange(max_seq_len):
			for blocks in xrange(block_size):
				x_data[examples][seqs][blocks] = seqs_x[examples][seqs][blocks]
				y_data[examples][seqs][blocks] = seqs_y[examples][seqs][blocks]
		print('Saved example ', (examples+1), 'of', nb_examples/2)

	#Might want to go a level deeper into the blocks
	#Bring a second corpus or library into tensor form
	#Run each block of generated sequence against each of second corpus
	#calculate difference in mean and/or variance (means gets precedence) 
	#(if mean and variance of any two+ blocks are same, they might be the same sound and 
	#   should be deleted)
	#The smallest difference block will be taken from corpus two and appended onto
	#output vector to be converted to wav and played and rated
	mean_x = np.mean(np.mean(x_data, axis=0), axis=0) #Mean across num examples and num timesteps
	std_x = np.sqrt(np.mean(np.mean(np.abs(x_data-mean_x)**2, axis=0), axis=0)) # STD across num examples and num timesteps
	std_x = np.maximum(1.0e-8, std_x) #Clamp variance if too tiny
	x_data[:][:] -= mean_x #Mean 0
	x_data[:][:] /= std_x #Variance 1
	y_data[:][:] -= mean_x #Mean 0
	y_data[:][:] /= std_x #Variance 1

	np.save(out_file+'_mean', mean_x)
	np.save(out_file+'_var', std_x)
	np.save(out_file+'_x', x_data)
	np.save(out_file+'_y', y_data)
	print('Done!')
	print(mean_x.shape, '\n', std_x.shape)

#Copied this code into bangermaker.py
def make_brain(block_size, timestep=seq_len):
	'''Want to rename seq_len, timestep'''
	print('Building brain...\n')
	model = Sequential()
	model.add(LSTM(block_size, input_shape=(seq_len, block_size), return_sequences=True))
	#model.add(Dropout(0.2))
	model.add(Dense(block_size))
	#model.add(Activation('linear'))
	return model

def train_brain(model, x_data, y_data, nb_epochs=1):
	print('Braining...\n')
	optimizer = RMSprop(lr=0.01)
	model.compile(loss='mse', optimizer='rmsprop')
	model.fit(x_data, y_data, batch_size=10000, epochs=nb_epochs, verbose=2)
	#Make it save weights
	return

def gimme_inspiration(seed_len, data_train):
	'''From GRUV'''
	#What if you increase seed len?
	nb_examples, seq_len = data_train.shape[0], data_train.shape[1]
	r = np.random.randint(data_train.shape[0])
	seed = np.concatenate(tuple([data_train[r+i] for i in xrange(seed_len)]), axis=0)
	#1 example by (# of examples) timesteps by (# of timesteps) frequencies
	inspiration = np.reshape(seed, (1, seed.shape[0], seed.shape[1]))
	return inspiration

def compose(x_data):
	'''Could add length of composition (roughly)'''
	print('Doing brain stuff...\n')
	generation = []
	muse = gimme_inspiration(1, x_data)
	for ind in xrange(1):
		preds = model.predict(muse)
		print(preds)
		print(len(preds), len(preds[0]), len(preds[0][0]))

	#After generator is fixed, call convert tensor_to_np
	#Then np (Which must be flattened, I think) to wav to play back
	#Then user may evaluate (Thumbs up/down)
	#If up, append the generated piece to the full list (call the functions to get it to tensor form)
	#Retrain model. If down, delete product and run through 1 or more epochs
	#Rinse and repeat. If training vector becomes too large (set arbitrarily for your hardware)
	#Delete oldest example from x and y before retraining.

def run():
	option = raw_input('Would you like to name the output file? (y/n)')
	if option is 'y'.lower():
		out_file = raw_input('Enter filename:')
	else:
		out_file = 'train_'

	block_size = 2048
	max_seq_len = 215
	music, rate = wav_to_np('./notstatic/Damiano_Baldoni_-_Ive_not_fear.wav')
	X, Y = load_corpus('./notstatic/Damiano_Baldoni_-_Ive_not_fear.wav', block_size)
	seqs_x, seqs_y = serialize_corpus(X, Y)



if __name__ == '__main__':
	out_file = 'train_'
	block_size = 2048
	max_seq_len = 215
	#music, rate = wav_to_np('./notstatic/Damiano_Baldoni_-_Ive_not_fear.wav')
	#print(music, len(music), music[0])
	X, Y = load_corpus('./notstatic/Damiano_Baldoni_-_Ive_not_fear.wav', block_size)
	seqs_x, seqs_y = serialize_corpus(X, Y)
	print(type(Y), type(Y[0]), type(Y[0][0]))

	nb_examples = len(seqs_x)
	nb_output_dims = block_size * 2
	output_shape = (nb_examples, max_seq_len, block_size)
	x_data = np.zeros(output_shape).astype('float32')
	y_data = np.zeros(output_shape).astype('float32')

	print('\nCalculating mean and variance and saving data\n')
	x_data = np.array(seqs_x)
	y_data = np.array(seqs_y)
	for examples in xrange(nb_examples/2):
		for seqs in xrange(max_seq_len):
			for blocks in xrange(block_size):
				x_data[examples][seqs][blocks] = seqs_x[examples][seqs][blocks]
				y_data[examples][seqs][blocks] = seqs_y[examples][seqs][blocks]
		print('Saved example ', (examples+1), 'of', nb_examples/2)

	#Might want to go a level deeper into the blocks
	mean_x = np.mean(np.mean(x_data, axis=0), axis=0) #Mean across num examples and num timesteps
	std_x = np.sqrt(np.mean(np.mean(np.abs(x_data-mean_x)**2, axis=0), axis=0)) # STD across num examples and num timesteps
	std_x = np.maximum(1.0e-8, std_x) #Clamp variance if too tiny
	x_data[:][:] -= mean_x #Mean 0
	x_data[:][:] /= std_x #Variance 1
	y_data[:][:] -= mean_x #Mean 0
	y_data[:][:] /= std_x #Variance 1

	np.save(out_file+'_mean', mean_x)
	np.save(out_file+'_var', std_x)
	np.save(out_file+'_x', x_data)
	np.save(out_file+'_y', y_data)
	print('Done!')
	#print(type(x_data), len(x_data), len(x_data[0]), len(x_data[0][0]))
	print(mean_x.shape, '\n', std_x.shape)

	print('Building brain...\n')
	model = Sequential()
	model.add(LSTM(block_size, input_shape=(max_seq_len, block_size), return_sequences=True))
	#model.add(Dropout(0.2))
	model.add(Dense(block_size))
	#model.add(Activation('linear'))

	print('Braining...\n')
	optimizer = RMSprop(lr=0.01)
	model.compile(loss='mse', optimizer='rmsprop')
	model.fit(x_data, y_data, batch_size=10000, epochs=1, verbose=2)

	print('Doing brain stuff...\n')
	prediction = []
	muse = gimme_inspiration(1, x_data)

	print('\n#############################\n')

	for ind in xrange(1):
		preds = model.predict(muse)
		print(preds)
		print(len(preds), len(preds[0]), len(preds[0][0]))



'''y_left = y_t[0:][0:][:1]
	y_right = y_t[0:][0:][1:2]
	y_np = np.asarray(y_left)
	y_right = np.asarray(y_right)
	y_np = np.column_stack((y_np, y_right))

	x_left = x_t[0:][0:][:1]
	x_right = x_t[0:][0:][1:2]
	print(x_right)
	x_np = np.asarray(x_left)
	x_right = np.asarray(x_right)
	x_np = np.column_stack((x_np, x_right))
	#print(y_np)'''