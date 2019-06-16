import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import numpy as np


def review_to_sentences(review, tokenizer, remove_stopwords=False):

    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            raw_sentence = review_to_wordlist(raw_sentence, remove_stopwords=True)
            sentences.append(raw_sentence.split())

    return sentences


#data clean
def review_to_wordlist(review, remove_stopwords=False):
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()

    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    # 5. Return a list of words
    return ' '.join(words)


def build_data_train_test(data_train, data_test, train_ratio=0.8):
    revs = []
    vocab = defaultdict(float)

    # Pre-process train data set
    for i in range(len(data_train)):
        rev = data_train[i]
        y = train['sentiment'][i]
        orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': y,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': int(np.random.rand() < train_ratio)}
        revs.append(datum)
    for i in range(len(data_test)):
        rev = data_test[i]
        orig_rev = ' '.join(rev).lower()
        words = set(orig_rev.split())
        for word in words:
            vocab[word] += 1
        datum = {'y': -1,
                 'text': orig_rev,
                 'num_words': len(orig_rev.split()),
                 'split': -1}
        revs.append(datum)

    return revs, vocab


def main():
    # load data
    train = pd.read_csv('./input/train.csv')
    test = pd.read_csv('./input/test.csv')
    sample_sub = pd.read_csv('./input/sample_submission.csv')
    clean_train = []
    for review in train["comment_text"]:
        clean_train.append(review_to_wordlist(review, remove_stopwords=True))

    clean_test = []
    for review in test["comment_text"]:
        clean_test.append(review_to_wordlist(review, remove_stopwords=True))

    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=5000)

    train_data_features = vectorizer.fit_transform(clean_train)
    x_train = train_data_features.toarray()
    test_data_features = vectorizer.transform(clean_test)
    x_test = test_data_features.toarray()
    print(x_test.shape)

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []  # Initialize an empty list of sentences
    for sentence in train["comment_text"]:
        sentences += review_to_sentences(review, tokenizer)
    for sentence in test["comment_text"]:
        sentences += review_to_sentences(review, tokenizer)




if __name__ == "__main__":
    main()
