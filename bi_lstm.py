from keras.models import Sequential
from keras.layers import Dense, GlobalMaxPool1D, Dropout, Embedding, Bidirectional, CuDNNLSTM


def bilstm_model(vocab_size, embedding_matrix):
    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length=300, weights=[embedding_matrix], trainable=False))
    model.add(Bidirectional(CuDNNLSTM(100, return_sequences=True)))
    model.add(GlobalMaxPool1D())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
