from keras.models import Sequential
from keras.layers import Dense, Embedding, CuDNNLSTM, Dropout


def lstm_model(vocab_size, embedding_matrix):
    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length=300, weights=[embedding_matrix], trainable=False))
    model.add(CuDNNLSTM(units=50))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(rate=0.15))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
