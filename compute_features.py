from collections import defaultdict
from collections import Counter
import cPickle as pickle
from nltk.corpus import wordnet as wn
import csv
import numpy as np

class FeatureExtractor:

  def __init__(self):
    self.arpabet = nltk.corpus.cmudict.dict()
    self.bigrams = defaultdict(int)
    self.unigrams = defaultdict(int)
    for line in open("w2_.txt", "r"):
      info = line.split()
      self.bigrams[(info[1], info[2])] = int(info[0])
    with open("unigram_freq.csv") as csvfile:
      r = csv.reader(csvfile, delimiter=',')
      for row in r:
        self.unigrams[row[0]] = row[1]

  # We expect jokes to have lower average bigram counts since the vocab and sentence
  # structure is often less standard
  def average_bigram_count(self, sentence):
    total = 0
    words = sentence.split()
    for i in range(len(words) - 1):
      total += self.bigrams[(words[i], words[i+1])]
    return total / float(len(words))


  # Weighs each synonym by the log of its unigram count. Intuition here is that we want
  # to measure whether we have any words with many common synonyms
  def synonym_measure(self, sentence):
    max_syn_strength = 0
    for word in sentence.split():
      syn_strength = 0
      synset = wn.synset(word)
      if synset:
        lemma = synset._lemma_names[0]
        if self.unigrams[lemma] > 0:
          syn_strength += len(synset) * np.log(self.unigrams[lemma])
      if syn_strength > max_syn_strength:
        max_syn_strength = syn_strength
    return max_syn_strength


  # Number of times the most common set of last 2 sounds occurs
  def rhyming(self, sentence):
    ending_counts = Counter()
    for word in sentence:
      ending_counts[self.arpabet[word][0][-2:]] += 1
    return ending_counts.most_common(1)[0][1]

  # Number of times the most common set of first 2 sounds occurs
  def alliteration(self, sentence):
    start_counts = Counter()
    for word in sentence:
      start_counts[self.arpabet[word][0][:-2]] += 1
    return start_counts.most_common(1)[0][1]



# https://www.cs.cmu.edu/~diyiy/docs/emnlp_yang_16.pdf <-- Remember to cite this paper for some of this stuff