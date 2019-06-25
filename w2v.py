from gensim.models import word2vec
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import os
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import codecs
lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()

APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}


def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    # Convert to lower case , so that Hi and hi are the same
    comment = comment.lower()
    # remove \n
    comment = comment.replace("\n", " ")
    comment = comment.replace("_", " ")
    # comment = re.sub("_", "", comment)
    # Split the sentences into words
    words = tokenizer.tokenize(comment)

    # (')aphostophe  replacement (ie)   you're --> you are
    # ( basic dictionary lookup : master dictionary present in a hidden block of code)
    words = [APPO[word] if word in APPO else word for word in words]
    words = [lem.lemmatize(word, "v") for word in words]
    # stops = set(stopwords.words("english"))
    # words = [w for w in words if not w in stops]
    clean_sent = " ".join(words)
    # remove any non alphanum,digit character
    clean_sent = re.sub("\W+"," ",clean_sent)
    clean_sent = re.sub("  "," ",clean_sent)
    return clean_sent

# def review_to_wordlist(review, remove_stopwords=True):
#     review_text = re.sub("[^a-zA-Z]", " ", review)
#     words = review_text.lower().split()
#     if remove_stopwords:
#         stops = set(stopwords.words("english"))
#         words = [w for w in words if not w in stops]
#
#     # 5. Return a list of words
#     return ' '.join(words)


def review_to_sentences(train, test):

    sentences = []
    for raw_sentence in train:
        if len(raw_sentence) > 0:
            raw_sentence = clean(raw_sentence)
            sentences.append(raw_sentence.split())
    for raw_sentence in test:
        if len(raw_sentence) > 0:
            raw_sentence = clean(raw_sentence)
            sentences.append(raw_sentence.split())
    return sentences


train_data = pd.read_csv('./input/train.csv')
train_text = train_data['comment_text']
train_text = np.array(train_text)
train_text = train_text.tolist()
test_data = pd.read_csv('./input/test.csv')
test_text = test_data['comment_text']
text_text = np.array(test_text)
ttest_text = test_text.tolist()

words = review_to_sentences(train_text, test_text)
print(len(words))
word_list = []

num_features = 300    # Word vector dimensionality
min_word_count = 1    # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

print("Training model")
model = word2vec.Word2Vec(words, workers=num_workers,
                          size=num_features,
                          min_count=min_word_count,
                          window=context,
                          sample=downsampling,
                          sg=0)

model.init_sims(replace=True)

model.save("300features_1minwords_10context.model")
print('finished')


with open('my_word_list.txt', 'w', encoding='utf-8') as f:
    for sentence in words:
        for word in sentence:
            if word not in word_list:
                word_list.append(word)
    tmp = '\n'.join(word_list)
    f.write(tmp)
print('finished writing word list')