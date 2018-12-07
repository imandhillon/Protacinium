
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

'''Relevant comments inside 'run()' method at bottom'''

def play_music(wavfile=''):
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

def write_np_as_wav(X, sample_rate=44100, filename='new.wav'):
	Xnew = X * 32767.0
	Xnew = Xnew.astype('int16')
	wav.write(filename, sample_rate, Xnew)
	return

def convert_sample_blocks_to_np_audio(blocks):
	song_np = np.concatenate(blocks)
	#song_np = [item for sublist in song_np for item in sublist]
	print(song_np, '\n><><><><><>')
	return song_np

def rand_wav():
	'''Example untrained output'''
	data = np.random.uniform(-1,1,44100) # 44100 random frequency samples between -1 and 1
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

	while num_samples < total_samples:
		block = music[num_samples:num_samples+block_size]
		if(block.shape[0] < block_size):
			padding = np.zeros((block_size - block.shape[0]))
			block = np.concatenate((block, padding))
		blocks.append(block)
		num_samples += block_size

	return blocks

def serialize_corpus(x_train, y_train, seq_len=215):
	seqs_x = []
	seqs_y = []
	cur_seq = 0
	total_seq = len(x_train)
	print('total seq: ', total_seq)
	print('max seq: ', seq_len)

	while cur_seq + seq_len < total_seq:
		seqs_x.append(x_train[cur_seq:cur_seq+seq_len])
		seqs_y.append(y_train[cur_seq:cur_seq+seq_len])
		cur_seq += seq_len

	return seqs_x, seqs_y

def make_tensors(file, seq_len=215, block_size=2048, out_file='train'):
	'''Have it handle directories *********'''
	music, rate = wav_to_np(file)
	try:
		music = music.sum(axis=1)/2
	except:
		pass

	x_t = np_to_sample(music, block_size)
	y_t = x_t[1:]					
	y_t.append(np.zeros(block_size)) 	
	seqs_x, seqs_y = serialize_corpus(x_t, y_t, seq_len)

	'''Pretty much taken from GRUV. '''
	nb_examples = len(seqs_x)

	print('\nCalculating mean and variance and saving data\n')
	x_data = np.array(seqs_x)
	y_data = np.array(seqs_y)

	for examples in xrange(nb_examples):
		for seqs in xrange(seq_len):
			for blocks in xrange(block_size):
				x_data[examples][seqs][blocks] = seqs_x[examples][seqs][blocks]
				y_data[examples][seqs][blocks] = seqs_y[examples][seqs][blocks]
		print('Saved example ', (examples+1), 'of', nb_examples)
	
	mean_x = np.mean(np.mean(x_data, axis=0), axis=0) #Mean across num examples and num timesteps
	std_x = np.sqrt(np.mean(np.mean(np.abs(x_data-mean_x)**2, axis=0), axis=0)) # STD across num examples and num timesteps
	std_x = np.maximum(1.0e-8, std_x) #Clamp variance if too tiny
	print('mean:', mean_x, '\n', 'std:', std_x)

	x_data[:][:] -= mean_x #Mean 0
	x_data[:][:] /= std_x #Variance 1
	y_data[:][:] -= mean_x #Mean 0
	y_data[:][:] /= std_x #Variance 1

	np.save(out_file+'_mean', mean_x)
	np.save(out_file+'_var', std_x)
	np.save(out_file+'_x', x_data)
	np.save(out_file+'_y', y_data)
	print('Done!')

	for x in xrange(2):
		print(x_data[x], '\n')
	for x in xrange(2):
		print(y_data[x], '\n')

	print('mean/std shape: ', mean_x.shape, '\n', std_x.shape)
	return x_data, y_data

def make_brain(timestep=215, block_size=2048):
	'''Can fiddle with Keras methods to try to get better results, quicker.'''
	print('Building brain...\n')
	model = Sequential()
	model.add(LSTM(block_size, input_shape=(timestep, block_size), return_sequences=True))
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
	return model

def gimme_inspiration(seed_len, data_train):
	'''From GRUV. Will change this when I understand how to mathematically
	generate 'good' music.'''
	nb_examples, seq_len = data_train.shape[0], data_train.shape[1]
	r = np.random.randint(data_train.shape[0])
	seed = np.concatenate(tuple([data_train[r+i] for i in xrange(seed_len)]), axis=0)
	#1 example by (# of examples) timesteps by (# of timesteps) frequencies
	inspiration = np.reshape(seed, (1, seed.shape[0], seed.shape[1]))
	return inspiration

def compose(model, x_data):
	'''Could add choice of length of composition (roughly)'''
	print('Doing brain stuff...\n')
	generation = []
	muse = gimme_inspiration(1, x_data)
	for ind in xrange(1):
		preds = model.predict(muse)
		print(preds)
		print(len(preds), len(preds[0]), len(preds[0][0]))
		generation.extend(preds)
	return generation

def run():
	out_file = 'train'
	'''					sample rate * clip len / seq_len '''
	block_size = 2700	# Around min # of samples for human to (begin to) percieve a tone at 16Hz
	seq_len = 215


	'''*****(pseudo-code)*****
	corpus = []
	for file in dir:
		if file.endswith(.wav):
			music, rate = wav_to_np(file)
			music = music.sum(axis=1)/2
			corpus.extend(music)'''
			
	x_data, y_data = make_tensors('./notstatic/danceoflife.wav', seq_len, block_size)


	model = make_brain(seq_len, block_size)
	model = train_brain(model, x_data, y_data)
	masterpiece = compose(model, x_data)

	#If(set to use sound_samples)			
	#Now grab corpus 2
	#increase block size (I'm thinking 5012) for both vectors. For now, leave it as is.
	#replace blocks from masterpiece with blocks from sound_samples
	#calculate similarity by getting variance for each block
	#Concatenate closest relative variance block from sound_samples to new list
	#For now, just get variance. Need to figure out how to get relative variance
	#Maybe to do with mean variance for overall piece. Will try to math later.

	masterpiece = convert_sample_blocks_to_np_audio(masterpiece[0]) #Not final, but works for now
	print(masterpiece) #			Should now be a flat list
	masterpiece = write_np_as_wav(masterpiece)
	play_music() # Seems to get stuck here (at least sometimes). Need some fix for this. I don't remember if the gui version has that problem...
	print('\n\nWas it a masterpiece (or at least an improvement)?')

	#Now user may evaluate (Thumbs up/down) or choose to exit app. Save weights for model before exit
	#If up, append the generated piece to the full list (call the functions to get it to tensor form)
	#Retrain model. If down, delete product and run through 1 or more epochs (let user pick #<10)
	#Rinse and repeat. If training vector (x_data) becomes too large (set arbitrarily for your hardware)
	#Delete oldest example from x and y before retraining.
	#Save weights for model each time. Naming models would become relevant when there are multiple
	#trained for various corpora. For now, just the main model. Handle saving model after we 
	#get it to update/retrain model.

	'''Add CNN classifier after converting from Keras to Tensorflow to use generative-adversarial model.
	'''

	return

if __name__ == '__main__':
	run()
