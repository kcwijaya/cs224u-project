import pickle
import numpy as np
import os

# Tune these params.
max_len = 70
embed_len = 100
embedding_path = 'vsmdata/glove.6B/glove.6B.'+ str(embed_len) + 'd.txt'


def glove2dict(src_filename):
    data = {}
    with open(src_filename, 'rb') as f:
        while True:
            try:
                line = next(f)
                line = line.strip().split()
                data[line[0].decode('utf-8')] = np.array(line[1: ], dtype=np.float)
            except StopIteration:
                break
            except UnicodeDecodeError:
                pass
    return data

def make_batches(embeds, batch_size):
    np.random.shuffle(embeds)
    batches = []
    i = 0
    while i*batch_size < embeds.shape[0]:
        if i == 0:
            batches.append(embeds[:batch_size])
        else:
            batches.append(embeds[int((i-1)*batch_size):int(i*batch_size)])
        i += 1
    return batches

def batch_data(data_filename, label_filename, batch_size):
    # This function takes in a batch size, a data filename, a
    # labels filename and returns a list of batches
    # where each batch list is a list of tuples of the form:
    # (padded_vec, mask, label)

    with open(data_filename, 'rb') as f:
        x = pickle.load(f)
    with open(label_filename, 'rb') as f:
        y = pickle.load(f)
    embeddings = []
    word_embeddings = glove2dict(embedding_path)
    padding_embed = np.array([0.0 for i in range(embed_len)])
    # print len(x)
    for i in range(len(x)):
        sen = x[i]
        words = sen.split(" ")
        if len(words) > max_len:
            continue
        embed_sen = []
        mask = []
        for w in  words: 
            if w not in word_embeddings:
                w = w.lower()
                w = w.replace('.', '')
                w = w.replace('?', '')
                w = w.replace('!', '')
                if w not in word_embeddings:
                    continue
            embed_sen.append(word_embeddings[w])
            mask.append(1)
        while len(embed_sen) < max_len:
            embed_sen.append(padding_embed)
            mask.append(0)
        embeddings.append((np.array(embed_sen), np.array(mask), y[i]))
    batches = make_batches(np.array(embeddings), batch_size)
    return batches

    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)

def make_batches_nn(X, mask, Y, batch_size):
    state = np.random.get_state()

    np.random.shuffle(X)
    np.random.set_state(state) 
    np.random.shuffle(mask)
    np.random.set_state(state) 
    np.random.shuffle(Y)


    batches = []
    i = 0
    while i*batch_size < X.shape[0]: 
        if i == 0: 
            batches.append((X[:batch_size], mask[:batch_size], Y[:batch_size]))
        else:
            batches.append((X[int((i-1)*batch_size):int(i*batch_size)], mask[int((i-1)*batch_size):int(i*batch_size)], Y[int((i-1)*batch_size):int(i*batch_size)]))
        i += 1
    return batches

# Specifically for the neural nets.. easier to have matrices instead
# of inidividual tuples
def batch_data_nn(data_filename, label_filename, batch_size):
    with open(data_filename, 'rb') as f:
        x = pickle.load(f, encoding='latin1')
    with open(label_filename, 'rb') as f:
        y = pickle.load(f, encoding='latin1')
    embeddings = []
    word_embeddings = glove2dict(embedding_path)
    padding_embed = np.array([0.0 for i in range(embed_len)])
    # print len(x)
    X = []
    Y = [] 
    X_mask = []
    actual_count = 0
    for i in range(len(x)):
        sen = x[i]
        words = sen.split(" ")
        if len(words) > max_len:
            continue
        embed_sen = []
        mask = []
        for w in  words: 
            if w not in word_embeddings:
                w = w.lower()
                w = w.replace('.', '')
                w = w.replace('?', '')
                w = w.replace('!', '')
                if w not in word_embeddings:
                    continue
            embed_sen.append(word_embeddings[w])
        used_words = len(embed_sen)
        embed_sen = embed_sen + [padding_embed] * (max_len-used_words)
        mask = [1] * used_words + [0] * (max_len-used_words)

        X.append(np.array(embed_sen))
        Y.append([1 if (y[i] == 1 and j == 1) or (y[i] == 0 and j == 0) else 0 for j in range(0, 2)])
        X_mask.append(np.array(mask))
        actual_count += 1

    X = np.array(X)
    X_mask = np.array(X_mask)
    Y = np.array(Y)
    X = np.reshape(X, (actual_count, max_len, embed_len))
    X_mask = np.reshape(X_mask, (actual_count, max_len))
    Y = np.reshape(Y, (actual_count, 2))
    batches = make_batches_nn(X, X_mask, Y, batch_size)
    return batches
