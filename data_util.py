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
                data[line[0]] = np.array(line[1: ], dtype=np.float)
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
    print len(x)
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
