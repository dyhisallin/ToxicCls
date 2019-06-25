from keras.layers import Embedding, Convolution1D, MaxPool1D, Flatten, Dense, Dropout
from keras.models import Sequential


def cnn_model(vocab_size, embedding_matrix):
    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length=300, weights=[embedding_matrix], trainable=False))
    model.add(Convolution1D(20, kernel_size=5, activation='relu', padding='valid'))
    model.add(MaxPool1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, ))
    model.add(Dense(6, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

