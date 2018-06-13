import numpy as np 
import tensorflow as tf 
from tensorflow.python.ops import embedding_ops 
from tensorflow.python.ops import variable_scope as vs 
import sklearn 
from data_util import batch_data_nn, split_batches
from models.util import Progbar 
import os

class Model: 
    def __init__(self, name, max_length=70, char_max_len=20, glove_path = 'data/glove/glove.6B.300d.txt', glove_dims=300):
        self.num_classes = 2
        self.max_length = max_length
        self.char_max_len=20
        self.session = tf.Session() 
        self.embedding_size = glove_dims
        self.name = name

        print('Building computation graph.')
        with tf.variable_scope("Model", initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True)):
            # Adds placeholders for x, y, and mask
            self.add_placeholders()

            # Builds the graph. Output is used for loss, prob_dist is used for getting the class
            self.output, self.predictions = self.build_graph()

            # Adds the loss operator (cross entropy)
            self.loss = self.get_loss(self.y_placeholder, self.output) 

            self.loss_summary = tf.summary.scalar("Loss", self.loss)
            
            # The training op - AdamOptimizer 
            self.train_op = self.optimizer(self.loss) 

            # self.prob_dist = tf.Print(self.prob_dist, ["prob dist", self.prob_dist, tf.shape(self.prob_dist)], summarize=200)

            # # Gets metrics. Right now, just the accuracy. 
            self.metrics = self.get_metrics(self.y_placeholder, self.predictions)
            self.metrics_summary = tf.summary.scalar("Accuracy", self.metrics)
            
        log_file = "./logs/" + name
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.writer = tf.summary.FileWriter(log_file, self.session.graph)        
        self.session.run(tf.global_variables_initializer()) 
        print('Done building computation graph.') 

    def build_graph(self): 
        raise NotImplemented('Must implement build_graph()')

    def add_placeholders(self):
        self.X_placeholder = tf.placeholder(tf.float32, [None, self.max_length, self.embedding_size], name='X_placeholder')
        self.X_mask_placeholder = tf.placeholder(tf.int32, [None, self.max_length], name='X_mask')
        self.y_placeholder = tf.placeholder(tf.int32, [None, self.num_classes], name='Y_placeholder')
        self.X_char_placeholder = tf.placeholder(tf.float32, [None, self.max_length, self.char_max_len, self.embedding_size], name='X_char_placeholder')
        self.Y_placeholder = tf.placeholder(tf.float32, [None, self.char_max_len, self.embedding_size], name='X_placeholder')
        self.features = tf.placeholder(tf.float32, [None, self.max_length, 4], name='Features')
        self.is_training = tf.placeholder(tf.bool, shape=())

    def add_embedding_layer(self, word_embeddings): 
        with vs.variable_scope("embeddings"):
            embedding_matrix = tf.constant(word_embeddings, dtype=tf.float32, name="emb_matrix")
            self.word_embeddings = embedding_ops.embedding_lookup(embedding_matrix, self.X_placeholder)

    def get_loss(self, y, output):
        raise NotImplemented('Must implement get_loss()')

    def optimizer(self, loss):
        raise NotImplemented('Must implement optimizer()') 

    def predict(self, X, chars, mask, y, batch_size=None): 
        raise NotImplemented('Must implement predict()')

    # The main train loop
    def fit(self, train_dataset, train_label_dataset, num_epochs=10):
        print('Training Model...')
        best_valid_loss = float('inf')
        best_valid_accuracy = float('-inf')
        batches, X_raw = batch_data_nn('combined_data.pickle', 'labels.pickle', self.batch_size)
        train_batches, (X_valid, X_char_valid,  X_mask_valid, y_valid, features_valid), (X_test, X_char_test, X_mask_test, y_test, features_test), X_raw = split_batches(batches,X_raw, 0.7, 0.2)

        print("Starting epochs.")
        counter = 0
        for epoch in range(1, num_epochs+1):
            progbar = Progbar(target = len(train_batches)) 

            print('Epoch #{0} out of {1}: '.format(epoch, num_epochs))
            for batch, (X_batch, X_char_batch, mask_batch, y_batch, features) in enumerate(train_batches):
                train_loss, _, train_metrics, loss_summary, metrics_summary = self.session.run((self.loss, self.train_op, self.metrics, self.loss_summary, self.metrics_summary), {
                    self.X_placeholder : X_batch, 
                    self.y_placeholder: y_batch, 
                    self.X_char_placeholder : X_char_batch, 
                    self.features: features,
                    self.X_mask_placeholder: mask_batch, 
                    self.is_training : True,
                })
                self.writer.add_summary(loss_summary, counter)
                self.writer.add_summary(metrics_summary, counter)
                progbar.update(batch+1, [('Train Loss', train_loss), ('Accuracy', train_metrics)])
                counter += 1
            print('Training Loss: {0:.4f} {1} Training Accuracy {2:.4f} {3}'.format(train_loss, '*', train_metrics, '*'))

            print("Epoch complete. Calculating validation loss...")

            print(self.get_stats_table(X_valid, X_char_valid, X_mask_valid, y_valid, features_valid))

        print("All Epochs complete. Calculating test loss...")

        print(self.get_stats_table(X_test, X_char_test, X_mask_test, y_test, features_test, X_raw))

        print("Done training.")

    def save(self, filename): 
        directory = os.path.dirname(filename) 
        if not os.path.exists(directory):
            os.makedirs(directory)

        tf.train.Saver().save(self.session, filename) 

    def load(self, filename):
        tf.train.Saver().restore(self.session, filename) 

    def get_metrics(self, y, predictions): 
        gold = tf.argmax(y, 1)
        correct = tf.equal(predictions, gold)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy

    def get_accuracies(self, predictions, y, sens=None):
        misclassified = None
        if sens is not None: 
            misclassified = []
            for i in range(0, predictions.shape[0]):
                if y[i] != predictions[i]: 
                    misclassified.append((sens[i], y[i]))
        return np.mean(y == predictions), misclassified   

    def get_precision_recall_f1(self, predictions, y):
        scores = {
            'precision': sklearn.metrics.precision_score(y, predictions),
            'recall': sklearn.metrics.recall_score(y, predictions),
            'f1': sklearn.metrics.f1_score(y, predictions),
        }
        return scores

    def save_file(self, missclassified):
        with open('missclassifed_' + self.name + '.txt', 'w') as f: 
            f.write("Total Missclassified: " + str(len(missclassified))+"\n")
            for i in range(0, len(missclassified)):
                f.write(str(missclassified[i][0]) + " " + str(missclassified[i][1]) + "\n")
        return

    def get_stats_table(self, X, chars, mask, y, features, X_raw=None):
        table  = '------------------------------------------------------------\n'
        table += '| Accuracy | Precision | Recall |   F1   |\n'
        table += '------------------------------------------------------------\n'  
    
        predictions = self.predict(X, chars, mask, y, features, self.batch_size)
        y = np.argmax(y, 1)
        accuracy, missclassified= self.get_accuracies(predictions, y, X_raw)
        if (missclassified is not None):
            self.save_file(missclassified)
        metrics = self.get_precision_recall_f1(predictions, y)

        row_format = '|  {0:.4f}  |  {1:.4f}  |  {2:.4f}  | {3:.4f} |\n'  
        row = [accuracy, metrics['precision'], metrics['recall'], metrics['f1']]
        table += row_format.format(*row)

        return table

