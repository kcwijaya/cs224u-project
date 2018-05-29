import numpy as np 
import tensorflow as tf 
from tensorflow.python.ops import embedding_ops 
from tensorflow.python.ops import variable_scope as vs 
from GloveProcessing import get_glove_embeddings
import sklearn 

class Model: 
    def __init__(self, max_length=70, glove_path = 'data/glove/glove.6B.100d.txt', glove_dims=100):
        self.num_classes = 2
        self.max_length = max_length
        self.session = tf.Session() 
        self.embedding_size = glove_dims

        # word_emb_matrix, word2id, id2word = get_glove_embeddings(glove_path, glove_dims)

        with tf.variable_scope("Model", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            # Adds placeholders for x, y, and mask
            self.add_placeholders()

            # Builds the graph. Output is used for loss, prob_dist is used for getting the class
            self.output, self.prob_dist = self.build_graph()

            # Adds the loss operator (cross entropy)
            self.loss = self.get_loss(self.y_placeholder, self.output) 

            # The training op - AdamOptimizer 
            self.train_op = self.optimizer(self.loss) 

            # Gets metrics. Right now, just the accuracy. 
            self.metrics = self.get_metrics(self.y_placeholder, self.prob_dist)

        self.session.run(tf.global_variables_initializer()) 
        print('Done.') 

    def build_graph(self): 
        raise NotImplemented('Must implement build_graph()')

    def add_placeholders(self):
        self.X_placeholder = tf.placeholder(tf.float32, [None, self.max_length, self.embedding_size], name='X_placeholder')
        self.X_mask_placeholder = tf.placeholder(tf.int32, [None, self.max_length], name='X_mask')
        self.y_placeholder = tf.placeholder(tf.int32, [None, self.num_classes], name='Y_placeholder')
        self.is_training = tf.placeholder(tf.bool, shape=())

    def add_embedding_layer(self, word_embeddings): 
        with vs.variable_scope("embeddings"):
            embedding_matrix = tf.constant(word_embeddings, dtype=tf.float32, name="emb_matrix")
            self.word_embeddings = embedding_ops.embedding_lookup(embedding_matrix, self.X_placeholder)

    def get_loss(self, y, output):
        raise NotImplemented('Must implement get_loss()')

    def optimizer(self, loss):
        raise NotImplemented('Must implement optimizer()') 

    def predict(self): 
        raise NotImplemented('Must implement predict()')

    def fit(self):
        raise NotImplemented('Must implement train()')

    def save(self, filename): 
        directory = os.path.dirname(filename) 
        if not os.path.exists(directory):
            os.makedirs(directory)

        tf.train.Saver().save(self.session, filename) 

    def load(self, filename):
        tf.train.Saver().restore(self.session, filename) 

    def get_metrics(self, y, predictions): 
        tf.Print(predictions, ['preds', predictions], summarize=200)
        misclassified = tf.not_equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
        return tf.reduce_mean(tf.cast(misclassified, tf.float32))

    def get_accuracies(self, predictions, y):
        misclassified = tf.not_equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
        return tf.not_equal(tf.cast(misclassified, tf.float32))

    def get_precision_recall_f1(self, predictions, y):
    	scores = []
    	for i in range(self.num_classes):
    		scores.append((
    			sklearn.metrics.precision_score(y[:, i], predictions[:, i]), 
    			sklearn.metrics.recall_score(y[:,i], predictions[:, i]), 
    			sklearn.metrics.f1_score(y[:, i], predictions[:, i]), 
    	))
    	return scores 

    def get_stats_table(self, X, Y, batch_size):
    	table  = '------------------------------------------------------------\n'
    	table += '| {0:^{1}} | Accuracy | Precision | Recall |   F1   |\n'.format('Class', 8)
    	table += '------------------------------------------------------------\n'  
    	classes = ["Joke", "Not Joke"]
    	predictions = self.predict(X, batch_size)
    	accuracies = self.get_accuracies(predictions, Y)
    	scores = self.get_precision_recall_f1(predictions,Y)
        
    	row_format = '| {0:<{5}} |  {1:.4f}  |   {2:.4f}  | {3:.4f} | {4:.4f} |\n'                       
    	for i in range(self.num_classes):
    		row = [classes[i], accuracies[i]] + list(scores[i])
    		table += row_format.format(*(row + [8]))   

    	final_row = ['Overall'] + [np.mean(accuracies)] + list(np.mean(scores, axis=0))
    	table += '------------------------------------------------------------\n'
    	table += row_format.format(*(final_row + [8]))                     
    	table += '------------------------------------------------------------\n'
    	return table

