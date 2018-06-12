import numpy as np 
import tensorflow as tf 
from models.Model import Model
from tensorflow.python.ops import rnn_cell 
from models.util import Progbar 
from data_util import batch_data_nn, split_batches, get_char_dict
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import DropoutWrapper
np.set_printoptions(threshold=np.nan)

class RNNModel(Model): 
	def __init__(self, name, depth = 5, lr = 0.001, max_length=70, kernel_size=5, filters=100, regularization_factor = 0.001, keep_prob = 0.5, batch_size=200, hidden_size=150):
		self.lr = lr
		self.regularization_factor = regularization_factor 
		self.keep_prob = keep_prob
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.filters=filters 
		self.kernel_size=kernel_size
		self.depth = depth 
		Model.__init__(self, name, max_length)

	def build_graph(self): 
		input_lens = tf.reduce_sum(self.X_mask_placeholder, axis=1) 
		inputs = self.X_placeholder

		char_embedding = self.convolve()
		inputs = tf.concat([self.X_placeholder, char_embedding], axis=-1)
		print(inputs.get_shape())
		print(self.features.get_shape())
		inputs = tf.concat([inputs, self.features], axis=-1)

		for i in range(0, self.depth): 
			lstm_cell_forward = rnn_cell.LSTMCell(self.hidden_size)
			lstm_cell_forward = DropoutWrapper(lstm_cell_forward, input_keep_prob=self.keep_prob)
			lstm_cell_backward = rnn_cell.LSTMCell(self.hidden_size)
			lstm_cell_backward = DropoutWrapper(lstm_cell_backward, input_keep_prob=self.keep_prob)
			(fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
					lstm_cell_forward, 
					lstm_cell_backward,
					inputs, 
					input_lens, 
					dtype=tf.float32, 
					scope='layer' + str(i))
			out = tf.concat([fw_out, bw_out], 2)
			out = tf.nn.dropout(out, self.keep_prob)
			inputs = out 

		h = tf.contrib.layers.fully_connected(out, num_outputs=self.hidden_size, activation_fn=tf.nn.relu)

		rows = tf.range(0, tf.shape(input_lens)[-1]) 
		indices = tf.subtract(input_lens, tf.ones_like(input_lens))
		indices = tf.nn.relu(indices)
		slicer = tf.stack([rows, indices], axis=1)
		
		h = tf.gather_nd(h, slicer) 

		weights = tf.get_variable("W", shape=[self.hidden_size, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
		bias = tf.get_variable("b", shape=[self.num_classes], initializer=tf.zeros_initializer())

		logits = tf.nn.xw_plus_b(h, weights, bias, name="logits")

		preds = tf.argmax(logits, 1)

		return logits, preds

	def convolve(self):
		embedding_matrix = self.X_char_placeholder
		embedding_matrix = tf.reshape(embedding_matrix, [-1, self.char_max_len, self.embedding_size])
		embedding_matrix = tf.nn.dropout(embedding_matrix, self.keep_prob) 

		conv = tf.layers.conv1d(
			inputs=embedding_matrix, 
			filters=self.filters,
			kernel_size=self.kernel_size, 
			padding="same", 
			activation=tf.nn.relu)

		conv = tf.layers.conv1d(
			inputs=embedding_matrix, 
			filters=self.filters, 
			kernel_size=self.kernel_size, 
			padding="same", 
			activation=tf.nn.relu) 

		conv = tf.reshape(conv, [-1, self.max_length, self.char_max_len, self.filters])
		conv = tf.reduce_max(conv, axis=2)
		return conv


	def masked_softmax(self, logits, dim):
	    """
	    Takes masked softmax over given dimension of logits.

	    Inputs:
	      logits: Numpy array. We want to take softmax over dimension dim.
	      mask: Numpy array of same shape as logits.
	        Has 1s where there's real data in logits, 0 where there's padding
	      dim: int. dimension over which to take softmax

	    Returns:
	      masked_logits: Numpy array same shape as logits.
	        This is the same as logits, but with 1e30 subtracted
	        (i.e. very large negative number) in the padding locations.
	      prob_dist: Numpy array same shape as logits.
	        The result of taking softmax over masked_logits in given dimension.
	        Should be 0 in padding locations.
	        Should sum to 1 over given dimension.
	    """
	    exp_mask = (1 - tf.cast(self.X_mask_placeholder, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
	    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
	    prob_dist = tf.nn.softmax(masked_logits, dim)

	    return masked_logits, prob_dist

	def get_loss(self, y, predictions):
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions, labels=tf.argmax(y, axis=1))

		loss = tf.reduce_mean(loss) 

		params = tf.trainable_variables()
		l2_loss = tf.add_n([tf.nn.l2_loss(p) for p in params if 'bias' not in p.name])
		regularization = l2_loss * self.regularization_factor

		return loss + regularization 


	def add_placeholders(self):
		self.regularization_factor = tf.placeholder_with_default(0.0, shape=())
		self.keep_prob = tf.placeholder_with_default(1.0, shape=())
		Model.add_placeholders(self)

	def optimizer(self, loss): 
		return tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

	def predict(self, X, chars, mask, y, features, batch_size=None): 
		if batch_size is None: batch_size = len(X) 

		preds = np.zeros((len(X)))
		for i in range(0, len(X), batch_size): 
			ret = self.session.run(self.predictions,
				{
					self.X_placeholder : X[i:i+batch_size], 
					self.X_char_placeholder : chars[i:i+batch_size],
					self.y_placeholder: y[i:i+batch_size], 
					self.X_mask_placeholder: mask[i:i+batch_size],
					self.features: features[i:i+batch_size],
					self.is_training : False
				})
			preds[i:i+batch_size] = ret
		return preds

	def save(self, filename): 
		directory = os.path.dirname(filename) 
		if not os.path.exists(directory):
			os.makedirs(directory)

		tf.train.Saver().save(self.session, filename) 

	def load(self, filename):
		tf.train.Saver().restore(self.session, filename) 


