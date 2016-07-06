# code repurposed from a keras example
import numpy as np

from keras import backend as K
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM, GRU, SimpleRNN
from keras.initializations import he_normal, glorot_uniform
from keras.layers import Convolution1D, MaxPooling1D
from keras.datasets import imdb


# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
nb_epoch = 2

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')

def ours(shape, activation='relu', back_term=True, drop_prev=0., drop_next=0.,
         c_forw=0.5, c_back=0.5, name=None, dim_ordering='th'):

    # E[f(z^l)^2], assumes input data is standardized
    energy_preserved = {'identity': 1., 'linear': 1., 'relu': 0.5,
                        'elu': 0.645, 'tanh': 0.394, 'custom': c_forw}
    # E[f'(z^(l+1))^2]
    back_correction = {'identity': 1., 'linear': 1., 'relu': 0.5,
                       'elu': 0.671, 'tanh': 0.216, 'custom': c_back}
    # overwrite c_forw and c_back parameters to match the activation; avoid with 'custom'
    c_forw, c_back = energy_preserved[activation], back_correction[activation]
    back_indicator = 1 if back_term else 0
    if drop_prev >= 0.99 or drop_next >= 0.99:
        raise Exception("Cannot dropout all neurons; dropout rate too high")

    s = np.sqrt(c_forw / (1 - drop_prev) + back_indicator * c_back * (1 - drop_next))

    if len(shape) == 2:
        W = np.random.normal(0, 1, size=shape)
        W /= np.linalg.norm(W, axis=0) + 1e-12
    elif len(shape) == 4:
        if dim_ordering == 'th':
            representations = []
            for _ in range(shape[0]):
                u = np.random.normal(size=np.array([1, shape[1], shape[2], shape[3]]))
                representations.append(u[:, :, :, :] / np.sqrt(np.sum(np.square(u[:, :, :, :])) + 1e-12))
            W = np.reshape(np.concatenate(representations, 0), (shape[0], shape[1], shape[2], shape[3]))
        else:
            raise Exception('Invalid dim_ordering: ' + dim_ordering)

    return K.variable(W / s, name=name)

def run(init='ours'):
    if init == 'ours':
        model = Sequential()
        model.add(Embedding(max_features, embedding_size, input_length=maxlen))
        model.add(Dropout(0.3))
        model.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=filter_length,
                                border_mode='valid',
                                activation='relu',
                                init=lambda shape, name='none': ours(shape, name=name,  drop_prev=0.3),
                                subsample_length=1))
        model.add(MaxPooling1D(pool_length=pool_length))
        model.add(LSTM(lstm_output_size,
                       init=lambda shape, name='none': ours(shape, name=name),
                       dropout_W=0.5))
        model.add(Dense(1, init=lambda shape, name='none': ours(shape, name=name, drop_prev=0.5)))
        model.add(Activation('sigmoid'))

    elif init == 'xavier':
        model = Sequential()
        model.add(Embedding(max_features, embedding_size, input_length=maxlen))
        model.add(Dropout(0.3))
        model.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=filter_length,
                                border_mode='valid',
                                activation='relu',
                                init=glorot_uniform,
                                subsample_length=1))
        model.add(MaxPooling1D(pool_length=pool_length))
        model.add(LSTM(lstm_output_size, dropout_W=0.5, init=glorot_uniform))
        model.add(Dense(1, init=glorot_uniform))
        model.add(Activation('sigmoid'))
    elif init == 'he':
        model = Sequential()
        model.add(Embedding(max_features, embedding_size, input_length=maxlen))
        model.add(Dropout(0.3))
        model.add(Convolution1D(nb_filter=nb_filter,
                                filter_length=filter_length,
                                border_mode='valid',
                                activation='relu',
                                init=he_normal,
                                subsample_length=1))
        model.add(MaxPooling1D(pool_length=pool_length))
        model.add(LSTM(lstm_output_size, dropout_W=0.5, init=he_normal))
        model.add(Dense(1, init=he_normal))
        model.add(Activation('sigmoid'))


    print('Training...')
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
    print(init, 'test score:', score)
    print(init, 'test accuracy:', acc)

for init in ['ours', 'xavier', 'he']:
    run(init)
# for init in ['ours']:
#     run(init)

