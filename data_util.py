import pickle
import numpy as np
import os
import sys 
from compute_features import FeatureExtractor
import random 

# Tune these params.
max_len = 70
char_max_len = 20
embed_len = 300
embedding_path = 'vsmdata/glove.6B/glove.6B.'+ str(embed_len) + 'd.txt'
char_embedding_path = 'vsmdata/char_embeddings.txt'

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

def make_batches_nn(X, X_char, mask, Y, features, batch_size, X_raw):
    # state = np.random.get_state()
    #
    # np.random.shuffle(X)
    # np.random.set_state(state) 
    # np.random.shuffle(mask)
    # np.random.set_state(state) 
    # np.random.shuffle(Y)
    # np.random.set_state(state)
    # np.random.shuffle(X_char)
    # np.random.set_state(state)
    # np.random.shuffle(features)
    # np.random.set_state(state)
    # np.random.shuffle(X_raw)
    #
    
    shuffled = list(zip(X, X_char, mask, Y, features, X_raw))
    random.shuffle(shuffled)
    X, X_char, mask, Y, features, X_raw = zip(*shuffled)
    X = np.array(X)
    X_char = np.array(X_char)
    mask = np.array(mask)
    Y = np.array(Y)
    features = np.array(features)
    X_raw = np.array(X_raw)
    
    batches = []
    i = 0
    print("Making batches...")
    while i*batch_size < X.shape[0]: 
        if i == 0: 
            batches.append((X[:batch_size], X_char[:batch_size], mask[:batch_size], Y[:batch_size], features[:batch_size]))
        else:
            start = int((i-1)*batch_size)
            end = int(i*batch_size)
            batches.append((X[start:end], X_char[start:end], mask[start:end], Y[start:end], features[start:end]))
        i += 1
    print("Done making batches.")
    return batches, X_raw

def get_features(sentence, extractor):
    rhyming = extractor.rhyming(sentence)
    alliteration = extractor.alliteration(sentence) 
    bigrams = extractor.average_bigram_count(sentence)
    synonyms = extractor.synonym_measure(sentence)

    # return [bigrams, synonyms, rhyming, alliteration]
    return [bigrams, alliteration, bigrams, synonyms]

# Specifically for the neural nets.. easier to have matrices instead
# of inidividual tuples
def batch_data_nn(data_filename, label_filename, batch_size):
    with open(data_filename, 'rb') as f:
        x = pickle.load(f, encoding='latin1')
    with open(label_filename, 'rb') as f:
        y = pickle.load(f, encoding='latin1')
    embeddings = []
    extractor = FeatureExtractor()

    word_embeddings = glove2dict(embedding_path)
    padding_embed = np.array([0.0 for i in range(embed_len)])
    # print len(x)
    X = []
    X_raw = []
    Y = [] 
    features = []
    X_mask = []
    actual_count = 0
    for i in range(len(x)):
        sen = x[i]
        sen = sen.replace("\n", " ")
        X_raw.append(sen)
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

        feats = get_features(sen, extractor)

        features_padding = np.array([0.0 for i in range(0, len(feats))])
        sen_features = [feats] + [features_padding] * (max_len-1)

        sen_features = np.array(sen_features)
        features.append(sen_features)

        X.append(np.array(embed_sen))
        y_val = []
        if y[i] == 1:
            y_val.append(0)
        else: 
            y_val.append(1)

        if y[i] == 0:
            y_val.append(0)
        else:
            y_val.append(1)
        Y.append(y_val)
        # Y.append([1 if (y[i] == 1 and j == 1) or (y[i] == 0 and j == 0) else 0 for j in range(0, 2)])
        X_mask.append(np.array(mask))
        actual_count += 1

    X = np.array(X)
    X_mask = np.array(X_mask)
    Y = np.array(Y)
    X = np.reshape(X, (actual_count, max_len, embed_len))
    features = np.array(features)
    X_char = get_char_embeddings(x)
    X_raw = np.array(X_raw)

    X_mask = np.reshape(X_mask, (actual_count, max_len))
    Y = np.reshape(Y, (actual_count, 2))
    batches, X_raw = make_batches_nn(X, X_char, X_mask, Y, features, batch_size, X_raw)
    return batches, X_raw

def get_char_embeddings(x):
    char_embeddings = glove2dict(char_embedding_path) 
    padding_embed = np.array([0.0 for v in range(embed_len)])
    X_char = [] 

    print("Getting char embedding for ", len(x))
    count = 0
    for sentence in x: 
        if (len(X_char) % 500 == 0):
            print("Done with ", len(X_char))

        sentence_embedding = [] 
        sentence = sentence.replace("\n", " ")

        words = sentence.split(" ")

        if len(words) > max_len: 
            continue

        chars = [[char for char in word] for word in words]

        bail_out = False
        for word in chars:
            word_embed = []

            if len(word) > char_max_len:
                bail_out = True
                break

            for w in word: 
                if w not in char_embeddings:
                    continue
                word_embed.append(char_embeddings[w])

            curr_len = len(word_embed)
            if curr_len < char_max_len: 
                word_embed = word_embed + [padding_embed for k in range(0, char_max_len-curr_len)]

            sentence_embedding.append(word_embed)

        if bail_out:
            continue

        for bah in sentence_embedding: 
            if (len(bah) != char_max_len):
                print(len(bah)) 

        curr_sen_len = len(sentence_embedding) 
        if curr_sen_len < max_len: 
            for i in range(0, max_len-curr_sen_len):
                new_word = [padding_embed for k in range(0, char_max_len)] 
                sentence_embedding.append(np.array(new_word)) 

        sentence_embedding = np.array(sentence_embedding)
        X_char.append(sentence_embedding) 

    # X_char = np.array(X_char)
    print("Done with char embeddings...")
    return X_char

def get_char_dict():
    return glove2dict(char_embedding_path)

def split_batches(batches, X_raw, train_split, valid_split):
    test_split = 1-train_split-valid_split
    total_batches = len(batches)
    num_train_batches = int(np.ceil(total_batches*train_split))
    num_valid_batches = int(np.ceil(total_batches*(train_split+valid_split)))
    num_test_batches = total_batches-num_valid_batches
    train_batches = batches[:num_train_batches]
    validation_set = batches[num_train_batches:num_valid_batches]
    test_set = batches[num_valid_batches:]
    X_raw = X_raw[num_valid_batches:]

    print("Data composition: train split", train_split, "("+ str(num_train_batches) + ")", "valid split", valid_split, "(" + str(num_valid_batches) + ")", "test split", test_split, "(" + str(num_test_batches) + ")")

    valid_batches = get_split(validation_set)
    test_batches = get_split(test_set)

    return train_batches, valid_batches, test_batches, X_raw

def get_split(dataset): 
    x_batches = []
    x_char_batches = []
    x_masks = [] 
    y_batches = [] 
    features_batches = []

    for batch, (x_batch, x_char_batch, x_mask, y_batch, features_batch) in enumerate(dataset):
        x_batches.append(x_batch) 
        x_masks.append(x_mask)
        x_char_batches.append(x_char_batch)
        y_batches.append(y_batch) 
        features_batches.append(features_batch)

    X = np.concatenate(x_batches, axis=0)
    X_mask = np.concatenate(x_masks, axis=0)
    X_char = np.concatenate(x_char_batches, axis=0)
    y = np.concatenate(y_batches, axis=0)
    features = np.concatenate(features_batches, axis=0)

    return (X, X_char, X_mask, y, features)
