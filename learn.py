from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
import nn_utils.network_utils as network_utils
import config.nn_config as nn_config

config = nn_config.get_neural_net_configuration()
input_file = config['model_file']
cur_iter = 0

model_basename = config['model_basename']
model_filename = model_basename + str(cur_iter)

print("Loading training data")

#load tensors
x_train = np.load(input_file + '_x.npy')
y_train = np.load(input_file + '_y.npy')
print(x_train)
print('Loaded data')

#find # of frequencies
freq_space_dims = x_train.shape[2]
hidden_dims = config['hidden_dimension_size']

model = network_utils.create_lstm_network(num_frequency_dimensions=freq_space_dims, num_hidden_dimensions=hidden_dims)

if os.path.isfile(model_filename):
	model.load_weights(model_filename)

num_iters = 50
epochs_per_iter = 20
batch_size = 1

print('Training')
while cur_iter < num_iters:
	print("Iteration: " + str(cur_iter))
	history = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs_per_iter, verbose=1, validation_split=0.0)
	cur_iter += epochs_per_iter

print('Trained')
model.save_weights(model_basename + str(cur_iter))