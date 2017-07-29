import tensormaker

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