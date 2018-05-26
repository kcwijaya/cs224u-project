import numpy as np 
import tensorflow as tf 

class Model: 
	def __init__(self):
		self.num_classes = 2 # Joke, Not-Joke 
		self.session = tf.Session() 

		self.output = self.build_graph() 
		self.loss = self.loss(self.y_placeholder, self.output) 
		self.train_op = self.optimizer(self.loss) 

		self.session.run(tf.global_variables_initializer()) 
		print('Done.') 

	def build_graph(self): 
		raise NotImplemented('Must implement build_graph()'); 

	def loss(self, y, output):
		raise NotImplemented('Must implement loss()');

	def predict(self): 
		raise NotImplemented('Must implement predict()');

	def train(self):
		raise NotImplemented('Must implement train()');

	def save(self, filename): 
		directory = os.path.dirname(filename) 
		if not os.path.exists(directory):
			os.makedirs(directory)

		tf.train.Saver().save(self.session, filename) 

	def load(self, filename):
		tf.train.Saver().restore(self.session, filename) 


