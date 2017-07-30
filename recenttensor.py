from __future__ import print_function
from random import randint
#from keras.backend import manual_variable_initialization manual_variable_initialization(True)
#Might need for saving weights/model

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
	'''From stackoverflow'''
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
	'''Need to flatten list. Banger comes in as [[[freqs0],[freqs_n], ...]]]
	if we want to choose song length, it would be 
		[[[freqs0],[freqs_n], ...], [freqs0],[freqs_n], ...]]]
	In the end of this method, must have form 
		[freqs]'''
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
	music = music.sum(axis=1)/2

	x_t = np_to_sample(music, block_size)
	y_t = x_t[1:]					#If x, then y. Sorta
	y_t.append(np.zeros(block_size)) 	#make it fit
	seqs_x, seqs_y = serialize_corpus(x_t, y_t)

	'''Pretty much taken from GRUV. '''
	nb_examples = len(seqs_x)

	#nb_output_dims = block_size * 2
	#output_shape = (nb_examples, max_seq_len, block_size)
	'''x_data = np.zeros(output_shape).astype('float32')
	y_data = np.zeros(output_shape).astype('float32')'''

	print('\nCalculating mean and variance and saving data\n')
	x_data = np.array(seqs_x)
	y_data = np.array(seqs_y) #		This is questionable...

	#Runs with half of examples to save time and memory
	for examples in xrange(nb_examples/2):
		for seqs in xrange(seq_len):
			for blocks in xrange(block_size):
				x_data[examples][seqs][blocks] = seqs_x[examples][seqs][blocks]
				y_data[examples][seqs][blocks] = seqs_y[examples][seqs][blocks]
		print('Saved example ', (examples+1), 'of', nb_examples/2)

	#Need to print stuff and see whats happening here.
	#----------Idea Time----------
	#Bring a second corpus or library into ndarray form and cut to blocks
	#Run each block of generated sequence against each of second corpus
	#calculate difference in mean and/or variance (means gets precedence) 
	#(if mean and variance of any two+ blocks are same, they might be the same sound and 
	#   should be deleted)
	#The smallest difference block will be taken from corpus two and appended onto
	#output vector to be converted to wav and played and rated

	#Will want to reshape so 'blocks' are, say 1/8 or 1/4 of a second ***
	#if the frequencies produced by one instrument are inherently higher or lower,
	#results would suck. Normalize frequencies? 
	#Add or subtract the difference in mean frequencies. 
	#****Mean is theoretically 0 because these are points of a soundwave.
	#Instead, find variance... make output vector closer to corpus 2 (...?...)
	#Leave corpus 2 alone, of course.
	#Should theoretically preserve rhythm of generation (based on corpus 1)
	#Then take the closest sound block found in corpus 2
	#So now you may produce music with the stylings of Bach and the sounds of Jimi Hendrix
	#The comments inside 'run' make more sense
	print('full corpus:', x_data, x_data.shape)
	for x in xrange(2):
		print(x_data[x], '\n')
	for x in xrange(2):
		print(y_data[x], '\n')
	
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

#Copied this code into bangermaker.py
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
	'''From GRUV'''
	#What if you increase seed len?
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
		generation.append(preds)
	return generation

def run():
	'''option = raw_input('Would you like to name the output file? (y/n):')
	if option is 'y'.lower():
		out_file = raw_input('Enter filename:')
	else:'''
	out_file = 'train'

	block_size = 2048
	seq_len = 215
	'''Select directory of corpus. Ask if want to use second directory
	for sound samples. (Experiment to improve results/do something cool)
	Would work best if genres are kept contained. Maybe later, handle m4a/mp3 to wav.
	Need this to work with gui, so Austin can handle this part.

	*****(pseudo-code)*****
	corpus = []
	for file in dir:
		if file.endswith(.wav):
			music, rate = wav_to_np(file)
			music = music.sum(axis=1)/2
			corpus.extend(music)
			'''

	x_data, y_data = make_tensors('./notstatic/Damiano_Baldoni_-_Ive_not_fear.wav')
	model = make_brain(seq_len, block_size)
	model = train_brain(model, x_data, y_data)
	banger = compose(model, x_data)
	#If(set to use sound_samples)
	#Now grab corpus 2
	#increase block size (I'm thinking 5012) for both vectors. For now, leave it as is.
	#replace blocks from banger with blocks from sound_samples
	#calculate similarity by getting variance for each block
	#Concatenate closest relative variance block from sound_samples to new list
	#For now, just get variance. Need to figure out how to get relative variance
	#Maybe to do with mean variance for overall piece. Will try to math later.
	print(banger, '**********\n', banger.shape, '\n')
	banger = convert_sample_blocks_to_np_audio(banger[0][0]) #Not final, but works for now
	print(banger) #			Should now be a flat list
	banger = write_np_as_wav(banger)
	play_music()
	print('\n\nWas this a banger (or at least an improvement)?')

	#		***This one next***
	#Now user may evaluate (Thumbs up/down) or choose to exit app. Save weights for model before exit
	#If up, append the generated piece to the full list (call the functions to get it to tensor form)
	#Retrain model. If down, delete product and run through 1 or more epochs (let user pick #<10)
	#Rinse and repeat. If training vector (x_data) becomes too large (set arbitrarily for your hardware)
	#Delete oldest example from x and y before retraining.
	#Save weights for model each time. Naming models would become relevant when there are multiple
	#trained for various corpora. For now, just the main model. Handle saving model after we 
	#get it to update/retrain model.
	return



if __name__ == '__main__':
	run()
