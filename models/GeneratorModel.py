import numpy as np 
import tensorflow as tf 
from tensorflow.python.ops import embedding_ops 
from tensorflow.python.ops import variable_scope as vs 
import sklearn 
from data_util import batch_data_nn, split_batches
from models.util import Progbar 
import os
from collections import Counter
import sys 
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import DropoutWrapper
np.set_printoptions(threshold=np.nan)
from tensorflow.python.ops import rnn_cell 

class GeneratorModel():
	def __init__(self, chars, max_length, jokes, vocab, char_to_idx, idx_to_char, sz, embed_size=300, depth = 5, lr = 0.001, sentence_length=70, regularization_factor = 0.001, keep_prob = 0.5, batch_size=100, hidden_size=512):
		self.chars = chars
		self.jokes = jokes 
		self.embed_size = embed_size
		self.vocab = vocab 
		self.session = tf.Session() 
		self.char_to_idx = char_to_idx 
		self.idx_to_char = idx_to_char
		self.sz = sz
		self.keep_prob = keep_prob 
		self.lr = lr 
		self.regularization_factor = regularization_factor 
		self.max_length = max_length 
		self.sentence_length = sentence_length
		self.depth = depth
		self.batch_size = batch_size
		self.hidden_size = hidden_size

		self.add_placeholders()

		# Builds the graph. Output is used for loss, prob_dist is used for getting the class
		self.output, self.predictions = self.build_graph()

		# Adds the loss operator (cross entropy)
		self.loss = self.get_loss(self.y, self.output) 

		# self.loss_summary = tf.summary.scalar("Loss", self.loss)

		# The training op - AdamOptimizer 
		self.train_op = self.optimizer(self.loss) 

		# self.prob_dist = tf.Print(self.prob_dist, ["prob dist", self.prob_dist, tf.shape(self.prob_dist)], summarize=200)

		# # Gets metrics. Right now, just the accuracy. 
		# self.metrics = self.get_metrics(self.y_placeholder, self.predictions)
		self.session.run(tf.global_variables_initializer()) 

	def add_placeholders(self): 
		self.x = tf.placeholder(tf.float32, [self.batch_size, self.max_length])
		self.y = tf.placeholder(tf.int32, [self.batch_size, self.max_length])
		# self.state = tf.placeholder_with_default(-1, [1, 1])
		self.sample = tf.placeholder_with_default(False, shape=())

	def build_graph(self):
		inputs = self.x
		inputs = tf.reshape(inputs, (-1, self.max_length, 1))

		# char_embedding = self.convolve()
		# inputs = tf.concat([self.x, char_embedding], axis=-1)
		# inputs = tf.concat([inputs, self.features], axis=-1)
		if self.sample:
			self.batch_size = 1 

		self.first_state = rnn_cell.LSTMCell.zero_state(self.batch_size, tf.float32) 
		# lstm_cell_backward = rnn_cell.LSTMCell(self.hidden_size)
		lstm_cell_forward = DropoutWrapper(self.first_state, input_keep_prob=self.keep_prob)
		lstm_cell_backward = rnn_cell.LSTMCell(self.hidden_size)
		lstm_cell_backward = DropoutWrapper(lstm_cell_backward, input_keep_prob=self.keep_prob)
		(fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
				lstm_cell_forward, 
				lstm_cell_backward,
				inputs, 
				dtype=tf.float32, 
				scope='layer' + str(i))
		out = tf.concat([fw_out, bw_out], 2)
		out = tf.layers.batch_normalization(out)
		out = tf.nn.dropout(out, self.keep_prob)
		inputs = out 

		lstm_cell_forward = rnn_cell.LSTMCell(self.hidden_size)
		lstm_cell_forward = DropoutWrapper(lstm_cell_forward, input_keep_prob=self.keep_prob)
		lstm_cell_backward = rnn_cell.LSTMCell(self.hidden_size)
		lstm_cell_backward = DropoutWrapper(lstm_cell_backward, input_keep_prob=self.keep_prob)
		(fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
				lstm_cell_forward, 
				lstm_cell_backward,
				inputs, 
				dtype=tf.float32, 
				scope='layer' + str(i))
		out = tf.concat([fw_out, bw_out], 2)
		out = tf.layers.batch_normalization(out)
		out = tf.nn.dropout(out, self.keep_prob)
		inputs = out 

		lstm_cell_forward = rnn_cell.LSTMCell(self.hidden_size)
		lstm_cell_forward = DropoutWrapper(lstm_cell_forward, input_keep_prob=self.keep_prob)
		lstm_cell_backward = rnn_cell.LSTMCell(self.hidden_size)
		lstm_cell_backward = DropoutWrapper(lstm_cell_backward, input_keep_prob=self.keep_prob)
		(fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
				lstm_cell_forward, 
				lstm_cell_backward,
				inputs, 
				dtype=tf.float32, 
				scope='layer' + str(i))
		out = tf.concat([fw_out, bw_out], 2)
		out = tf.layers.batch_normalization(out)
		out = tf.nn.dropout(out, self.keep_prob)
		inputs = out 

		lstm_cell_forward = rnn_cell.LSTMCell(self.hidden_size)
		lstm_cell_forward = DropoutWrapper(lstm_cell_forward, input_keep_prob=self.keep_prob)
		lstm_cell_backward = rnn_cell.LSTMCell(self.hidden_size)
		lstm_cell_backward = DropoutWrapper(lstm_cell_backward, input_keep_prob=self.keep_prob)
		(fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
				lstm_cell_forward, 
				lstm_cell_backward,
				inputs, 
				dtype=tf.float32, 
				scope='layer' + str(i))
		out = tf.concat([fw_out, bw_out], 2)
		out = tf.layers.batch_normalization(out)
		out = tf.nn.dropout(out, self.keep_prob)
		inputs = out 
			

		self.last_state = tf.contrib.layers.fully_connected(out, num_outputs=self.hidden_size, activation_fn=tf.nn.relu)
		h = tf.reshape(self.last_state, (-1, self.hidden_size))
		weights = tf.get_variable("W", shape=[self.hidden_size, self.sz], initializer=tf.contrib.layers.xavier_initializer())
		bias = tf.get_variable("b", shape=[self.sz], initializer=tf.zeros_initializer())

		logits = tf.nn.xw_plus_b(h, weights, bias, name="logits")	
		logits = tf.reshape(logits, (-1, self.max_length, self.sz))
		preds = tf.nn.softmax(logits)	
		return logits, preds

	def get_loss(self, y, logits):
		# logits = tf.cast(logits, tf.float32)
		# y = tf.reshape(y, -1)
		# logits = tf.reshape(logits, -1)
		# print(y.get_shape())
		# print(logits.get_shape())
		# loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
		# return loss	
		print(logits.get_shape())
		loss = tf.contrib.seq2seq.sequence_loss(logits, y, tf.layers.flatten(tf.ones_like(self.x)))
		return loss

	def encode(self, sentence):
		words = sentence.split(" ")
		ret = []
		for word in words: 
			ret.append(self.word_to_idx[word])
		return ret

	def decode(self, sentence):
		words = sentence.split(" ")
		ret = []
		for word in words: 
			ret.append(self.idx_to_word[word])
		return ret 

	def optimizer(self, loss): 
		return tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

	def get_batches(self, X):
		Y = X.copy() 
		Y = X[1:]
		Y = np.insert(Y, Y.shape[0]-1, X[0])

		print(X[0:5])
		print(Y[0:5])


		x_seq = np.array(np.array_split(X, np.arange(self.max_length, X.shape[0], self.max_length))[:-1])
		y_seq = np.array(np.array_split(Y, np.arange(self.max_length, Y.shape[0], self.max_length))[:-1])

		total_batches = int(np.floor(float(len(x_seq))/self.batch_size))
		print("Total batches", total_batches)

		x_batches = np.array_split(x_seq, np.arange(self.batch_size, x_seq.shape[0], self.batch_size))
		y_batches = np.array_split(y_seq, np.arange(self.batch_size, y_seq.shape[0], self.batch_size))
		x_batches = np.array(x_batches)
		y_batches = np.array(y_batches)
		# num_train_batches = int(np.ceil(len(batches)*0.7))
		# num_valid_batches = int(np.ceil(len(batches)*0.9))
		# num_test_batches = len(batches)-num_valid_batches
		# t_x = batches[:num_train_batches]
		# v_x = batches[num_train_batches:num_valid_batches]
		# te_x = batches[num_valid_batches:]
		validation_set, test_set = None, None
		# t_x = batches
		# t_y = np.copy(t_x) 
		# # v_y = np.copy(v_x)
		# # te_y = np.copy(te_x) 

		# t_y = t_y[1:]
		# t_y = [t_x[0]] + t_y
		# print(t_x[0:10])
		# print(t_y[0:10])


		# v_y[:-1] = v_x[1:]
		# v_y[-1] = v_x[0]

		# te_y[:-1] = te_x[1:]
		# te_y[-1] = te_x[0]	

		train_batches = list(zip(x_batches, y_batches))
		# validation_set = (v_x, v_y)
		# test_set = (te_x, te_y)	
		# print(x_batches[0][0])

		# print(y_batches[0][0])

		return train_batches

	# def predict(self, X, Y): 
	# 	preds = np.zeros((len(X)))
	# 	for i in range(0, len(X), self.batch_size): 
	# 		ret = self.session.run(self.predictions,
	# 			{
	# 			self.x : X[i:i+batch_size], 
	# 			self.y : Y[i:i+batch_size]
	# 			})
	# 		preds[i:i+self.batch_size] = ret
	# 	return preds

	# def calc_loss
	def train(self, X, num_epochs=5): 
		train_batches = self.get_batches(X)
		# print("Train batches!")
		# print(train_batches[1])

		for epoch in range(1, num_epochs+1):
			progbar = Progbar(target = len(train_batches)) 

			print('Epoch #{0} out of {1}: '.format(epoch, num_epochs))
			for batch, (x, y) in enumerate(train_batches):
				train_loss, _ = self.session.run([self.loss, self.train_op], {self.x: x, self.y: y})

				progbar.update(batch+1, [('Train Loss', train_loss)])
			print('Training Loss: {0:.4f} {1}'.format(train_loss, '*'))


		print("All Epochs complete. Sampling.")
		joke = self.sample(500)
		print(joke)

		# print(self.get_stats_table(X_test, X_char_test, X_mask_test, y_test, features_test))

		print("Done training.")

	def sample(self, length): 
		first = random.choice(self.vocab)
		x = [self.char_to_idx[first]]
		x = np.array(x) 
		state = self.session.run([self.first_state], {self.sample:True} )
		[state] = self.session.run([self.last_state], self.x: x, self.first_state: init_state)

		joke = first
		curr_char = first
		for it in range(0, length): 
			x = [char_to_idx[curr_char]]
			x = np.array(x) 

			[logits, state] = self.session([self.preds, self.last_state], {self.x: x, self.first_state: state})
			preds = logits[0] 
			pred = np.argmax(preds) 
			joke += self.idx_to_char(pred)
			curr_char = sel.idx_to_char(pred)
		print(joke)
		return joke

