import os
import scipy.io.wavfile as wav
import numpy as np
from pipes import quote
from config import nn_config
from parse_files import *

def wav_to_nptensor(dir, block_size, max_seq_len, out_file, useTimeDomain=False):
	files = []
	for file in os.listdir(dir):
		if file.endswith('.wav'):
			files.append(dir+file)

	x_chunks = []
	y_chunks = []
	num_files = len(files)
	for idx in xrange(num_files):
		f = files[idx]
		print 'Processing: ', (idx+1), '/', num_files
		print 'Filename: ', f
		#end up with fft blocks
		X, Y = load_training_ex(f, block_size, useTimeDomain=useTimeDomain)
		cur_seq = 0
		total_seq = len(X)
		print total_seq
		print max_seq_len
		while cur_seq + max_seq_len < total_seq:
			x_chunks.append(X[cur_seq:cur_seq+max_seq_len])
			y_chunks.append(X[cur_seq:cur_seq+max_seq_len])
			cur_seq += max_seq_len
	num_ex = len(x_chunks)
	num_dims_out = block_size * 2
	if(useTimeDomain):
		num_dims_out = block_size
	out_shape = (num_ex, max_seq_len, num_dims_out)
	x_data = np.zeros(out_shape)
	y_data = np.zeros(out_shape)
	for n in xrange(num_ex):
		for i in xrange(max_seq_len):
			x_data[n][i] = x_chunks[n][i]
			y_data[n][i] = y_chunks[n][i]
		print 'Saved example ', (n+1), ' / ', num_ex
	print 'Flushing to disk...'
	mean_x = np.mean(np.mean(x_data, axis=0), axis=0)
	std_x = np.sqrt(np.mean(np.mean(np.abs(x_data-mean_x)**2, axis=0), axis=0))
	std_x = np.maximum(1.0e-8, std_x)
	x_data[:][:] -= mean_x
	x_data[:][:] /= std_x
	y_data[:][:] -= mean_x
	y_data[:][:] /= std_x
	print(x_data)
	np.save(out_file+'_mean', mean_x)
	np.save(out_file+'_var', std_x)
	np.save(out_file+'_x', x_data)
	np.save(out_file+'_y', y_data)
	print 'Done!'


def load_training_ex(filename, block_size=2048, useTimeDomain=False):
	data, bitrate = read_wav_as_np(filename)
	print(block_size)
	print("****")            #########################
	x_t = np_audio_to_sample_blocks(data, block_size)
	y_t = x_t[1:]
	y_t.append(np.zeros(block_size))
	if useTimeDomain:
		return x_t, y_t
	x = time_blocks_to_fft(x_t)
	y = time_blocks_to_fft(y_t)
	return x, y

def read_wav_as_np(filename):
	rate, data = wav.read(filename)
	np_arr = data[1].astype('float32') / 32767.0 #Normalize 16-bit to [-1, 1] range
	return np_arr, data[0]

def np_audio_to_sample_blocks(song_np, block_size):
	block_lists = []
	total_samples = song_np.shape[0]
	#print(song_np.shape[0])
	num_samples = 0
	while(num_samples < total_samples):
		block = song_np[num_samples:num_samples+block_size]
		#print(block_size)
		#print("blk------")
		print(block.shape[0])#####################################
		#print(block.shape[1])
		print("pad-------")
		if(block.shape[0] < block_size):
			padding = np.zeros((block_size - block.shape[0]))
			print(padding.shape[0])                    ##################
			block = np.concatenate((block, padding), axis=0)
		block_lists.append(block)
		num_samples += block_size
	return block_lists

def time_blocks_to_fft(blocks_time_domain):
	fft_blocks = []
	for block in blocks_time_domain:
		fft_block = np.fft.fft(block)
		new_block = np.concatenate((np.real(fft_block), np.imag(fft_block)))
		fft_blocks.append(new_block)
	return fft_blocks


input_dir = './datasets/bangers/'
out_file = './datasets/model'
sampling_freq = 44100
hidden_dimension_size = 1024

clip_len = 10    #secs
block_size = sampling_freq/4   #size of input state
max_seq_len = int(round((sampling_freq*clip_len)/block_size))#for zero-padding song sequences

wav_to_nptensor(input_dir, block_size, max_seq_len, out_file)