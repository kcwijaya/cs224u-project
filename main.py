import os 
import io 
import sys 
import tensorflow as tf 
from models.RNNModel import RNNModel 
import pickle 
from data_util import glove2dict, make_batches, batch_data

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments") # relative path of experiments dir

tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.2, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_float("regularization_factor", 0.00001, "Regularize parameters by adding params l2 norm times this factor to loss.")
tf.app.flags.DEFINE_string("improvement", "", "Available improvements: cnn")

tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use")
tf.app.flags.DEFINE_integer("hidden_size", 75, "Size of the hidden states")
tf.app.flags.DEFINE_integer("max_length", 200, "The maximum context length of your model")
tf.app.flags.DEFINE_string("output_state_dimensions", "100", "Dimension of the output states h_i") # Added for CNN

# How often to print, save, eval
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 500, "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("eval_every", 500, "How many iterations to do per calculating loss/f1/em on dev set. Warning: this is fairly time-consuming so don't do it too often.")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).")

# Reading and saving data
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to save the model parameters and other info. Defaults to experiments/{experiment_name}")
tf.app.flags.DEFINE_string("glove_path", "", "Path to glove .txt file. Defaults to data/glove.6B.100d.txt")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.app.flags.DEFINE_string("experiment", "", "Experiment")
FLAGS = tf.app.flags.FLAGS 

def initialize_model(session, model, train_dir):
	session.run(tf.global_variables_initializer())

def main():
	print(FLAGS.experiment)
	model = RNNModel(name=FLAGS.experiment)
	
	# Right now, not using any of the flags. This just runs two epochs. Sorta messy rn b/c
	# we are dividing into validation set within the fit function instead of doing the split
	# beforehand.
	model.fit('combined_data.pickle', 'labels.pickle', num_epochs = 1)
	# for i in range(0, 20):
	# 	text = model.generate_text(70)
	# 	print(text)
	# 	print("\n\n")
	

if __name__ == '__main__':
	main()

