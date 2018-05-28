import numpy as np
import argparse
import pickle
import os
import sys 
import csv 
from sklearn.svm import SVC

# Tune these params....
test_percentage = 0.05
n_folds = 5
max_len = 70
data_embed_path = 'combined_data_embeddings_ml' + str(max_len) + '.pickle'
embed_len = 100
embedding_path = 'vsmdata/glove.6B/glove.6B.'+ str(embed_len) + 'd.txt'

def glove2dict(src_filename):
    data = {}
    with open(src_filename, 'rb') as f:
        while True:
            try:
                line = next(f)
                line = line.strip().split()
                data[line[0]] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data

def filter_data(x, y):
    if os.path.isfile(data_embed_path):
        with open(data_embed_path, 'rb') as f:
            return pickle.load(f)
    # Idea pickle this shit so only have to do it once...
    embeddings = []
    labels = []
    word_embeddings = glove2dict(embedding_path)
    padding_embed = np.array([0.0 for i in range(embed_len)])
    for i in range(len(x)):
        sen = x[i]
        words = sen.split(" ")
        if len(words) > max_len:
            continue
        embed_sen = []
        for w in  words: 
            if w not in word_embeddings:
                w = w.lower()
                w = w.replace('.', '')
                w = w.replace('?', '')
                w = w.replace('!', '')
                if w not in word_embeddings:
                    print(w)
                    continue
            embed_sen.append(word_embeddings[w])
        while len(embed_sen) < max_len:
            embed_sen.append(padding_embed)
        embeddings.append(np.array(embed_sen).flatten())
        labels.append(y[i])
    return np.array(embeddings), labels

def get_data(name, label):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    with open(label, 'rb') as f:
        labels = pickle.load(f)
    data, labels = filter_data(data, labels)
    comb_data = np.array([(data[i], labels[i]) for i in range(len(data))])
    np.random.shuffle(comb_data)
    data = comb_data
    test_x = [p[0] for p in data[:int(len(data) * test_percentage)]]
    train_x = [p[0] for p in data[int(len(data) * test_percentage) : ]]
    test_y = [p[1] for p in data[:int(len(data) * test_percentage)]]
    train_y = [p[1] for p in data[int(len(data) * test_percentage) : ]]
    return train_x, test_x, train_y, test_y

model_choice = {'rnn' : None,
                'svm' : SVC}

def train_model(model, X, y):
    print "Training Model"
    model.fit(X, y)
    return model

def test_model(model, test_X, test_y):
    print "Testing Model"
    correct = 0.0
    for i in range(len(test_X)):
        y = model.predict(test_X[i].reshape(1, -1))
        if y == test_y[i]:
            correct += 1.0
    print("Test accuracy is: " + str(float(correct / len(test_X))))

def run_model(args):
    train_X, test_X, train_y, test_y = get_data(args.d, args.l)
    model = model_choice[args.m]()
    model = train_model(model, train_X, train_y)
    test_model(model, test_X, test_y)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help='Select Model', default = 'rnn')
    parser.add_argument('-d', help='Specify Dataset Path', default = 'combined_data.pickle')
    parser.add_argument('-l', help='Specify Label Path', default = 'labels.pickle')
    return parser.parse_args()


# Cross validate
def main():
    args = parse_args()
    run_model(args)
    

if __name__ == "__main__":
    main()
