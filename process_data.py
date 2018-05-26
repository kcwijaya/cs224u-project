import cPickle as pickle

def main():
  joke_data = pickle.load(open("humorous_oneliners.pickle", "rb"))
  wiki_data = pickle.load(open("wiki_sentences.pickle", "rb"))
  labels = [0] * len(joke_data) + [1] * len(wiki_data)

  pickle.dump(joke_data + wiki_data, open("combined_data.pickle", "wb"))
  pickle.dump(labels, open("labels.pickle", "wb"))

if __name__ == '__main__':
  main()