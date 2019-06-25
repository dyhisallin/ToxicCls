import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import time
from gensim.models import KeyedVectors
w2v_model = KeyedVectors.load('300features_1minwords_10context.model')

TEXT_COLUMN = 'comment_text'
TARGET_COLUMN = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
start_time = time.time()
train_data = pd.read_csv('./input/train.csv', encoding='utf-8')
# train_data = pd.read_csv('./output/eda_train.csv', encoding='utf-8')
tmp_time = time.time() - start_time
print('read train data spent ' + str(tmp_time))
train_labels = train_data[TARGET_COLUMN].values
test_data = pd.read_csv('./input/test.csv', encoding='utf-8')
tmp_time = time.time() - start_time
print('read test data spent ' + str(tmp_time))

embedding_values = {}
for key, value in w2v_model.wv.vocab.items():
    word = str(key)
    coef = np.array(w2v_model[word], dtype='float32')
    embedding_values[word] = coef
tmp_time = time.time() - start_time
print('read w2v spent ' + str(tmp_time))

all_embs = np.stack(embedding_values.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()
emb_mean, emb_std

# mapping text to index
train_text = train_data['comment_text']
token = Tokenizer(num_words=200000, filters='#$%&()*+,-./:;<=>?@[\]^_`{|}~ ', lower=True)
token.fit_on_texts(train_text)
train_seq = token.texts_to_sequences(train_text)
train_pad_seq = pad_sequences(train_seq, maxlen=300, padding='post', truncating='post')

test_text = test_data['comment_text']
test_seq = token.texts_to_sequences(test_text)
test_pad_seq = pad_sequences(test_seq, maxlen=300, padding='post', truncating='post')

vocab_size = len(token.word_index) + 1
print('vocab size is '+str(vocab_size))

# get embedding_matrix ready
embedding_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, 300))
for word, i in tqdm(token.word_index.items()):
    values = embedding_values.get(word)
    if values is not None:
        embedding_matrix[i] = values
tmp_time = time.time() - start_time
print('get embedding_matrix ready spent ' + str(tmp_time))

from bi_lstm import bilstm_model
from cnn import cnn_model
from lstm import lstm_model


# model = cnn_model(vocab_size=vocab_size, embedding_matrix=embedding_matrix)
# model = lstm_model(vocab_size=vocab_size, embedding_matrix=embedding_matrix)
model = bilstm_model(vocab_size=vocab_size, embedding_matrix=embedding_matrix)
print(model.summary())

model.fit(x=train_pad_seq, y=train_labels, epochs=16, batch_size=64, validation_split=0.1)
tmp_time = time.time() - start_time
print('training model1 spent ' + str(tmp_time))
# model.save_weights('./output/bilstm_attention_model.h5')

predict = model.predict(test_pad_seq)
tmp_time = time.time() - start_time
print('predicting spent ' + str(tmp_time))

sample_submission = pd.read_csv('./input/sample_submission.csv', encoding='utf-8')
sample_submission[TARGET_COLUMN] = predict
sample_submission.to_csv('./output/w2v_bilstm_submission.csv', index=False)
print('finished')

