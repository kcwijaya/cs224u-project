import numpy as np 
import tensorflow as tf 
from tensorflow.python.ops import embedding_ops 
from tensorflow.python.ops import variable_scope as vs 
import sklearn 

class Model: 
    def __init__(self, max_length=70, glove_path = 'data/glove/glove.6B.100d.txt', glove_dims=100):
        self.num_classes = 2
        self.max_length = max_length
        self.session = tf.Session() 
        self.embedding_size = glove_dims

        with tf.variable_scope("Model", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            # Adds placeholders for x, y, and mask
            self.add_placeholders()

            # Builds the graph. Output is used for loss, prob_dist is used for getting the class
            self.output, self.predictions = self.build_graph()

            # Adds the loss operator (cross entropy)
            self.loss = self.get_loss(self.y_placeholder, self.output) 

            # The training op - AdamOptimizer 
            self.train_op = self.optimizer(self.loss) 

            # self.prob_dist = tf.Print(self.prob_dist, ["prob dist", self.prob_dist, tf.shape(self.prob_dist)], summarize=200)

            # # Gets metrics. Right now, just the accuracy. 
            self.metrics = self.get_metrics(self.y_placeholder, self.predictions)
            
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

    def predict(self, X, mask, y, batch_size=None): 
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
        # preds = tf.argmax(predictions, 1)
        # preds = tf.Print(preds, ['preds classes', preds], summarize=200)
        # true = tf.argmax(y, 1)
        # true = tf.Print(true, ['true classes', true], summarize=200)
        # predictions = tf.Print(predictions, ["preds", predictions, tf.shape(predictions)], summarize=self.batch_size)
        # y = tf.Print(y, ["classes", tf.argmax(y, 1), tf.shape(tf.argmax(y, 1))], summarize=self.batch_size)
        gold = tf.argmax(y, 1)
        correct = tf.equal(predictions, gold)
        # missclassified = tf.Print(misclassified, ['misclassified', misclassified], summarize=200)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        # precision = tf.metrics.precision(gold, predictions)
        # recall = tf.metrics.recall(gold, predictions)
        # f1 = sklearn.metrics.f1_score(gold, predictions)

        return accuracy

    # def get_accuracies(self, predictions, y):
    #     misclassified = tf.not_equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    #     return tf.not_equal(tf.cast(misclassified, tf.float32))

    def get_accuracies(self, predictions, y):
        return np.mean(y == predictions)    

    def get_precision_recall_f1(self, predictions, y):
        scores = {
            'precision': sklearn.metrics.precision_score(y, predictions),
            'recall': sklearn.metrics.recall_score(y, predictions),
            'f1': sklearn.metrics.f1_score(y, predictions),
        }
        return scores


    def get_stats_table(self, X, mask, y):
        table  = '------------------------------------------------------------\n'
        table += '| Accuracy | Precision | Recall |   F1   |\n'
        table += '------------------------------------------------------------\n'  
    
        predictions = self.predict(X, mask, y)
        y = np.argmax(y, 1)
        accuracy = self.get_accuracies(predictions, y)
        metrics = self.get_precision_recall_f1(predictions, y)

        row_format = '|  {0:.4f}  |  {1:.4f}  |  {2:.4f}  | {3:.4f} |\n'  
        row = [accuracy, metrics['precision'], metrics['recall'], metrics['f1']]
        table += row_format.format(*row)   

        return table

