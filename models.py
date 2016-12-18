import _pickle as cPickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling1D
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
from features import dumpFeatures

dumpFeatures(full=True, n_gram=False, pretrained=True, nb_words=None, namefile='pretrained_features.dat')

[X_train, y, X_test, max_features, W] = cPickle.load(open("pretrained_features.dat", "rb"))

np.random.seed(1)

model = Sequential()
model.add(Embedding(max_features+1, 200, input_length=X_train.shape[1], weights=[W]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1)
train_1 = model.predict_proba(X_train, batch_size=128)
test_1 = model.predict_proba(X_test)

cPickle.dump(train_1, open('train_conv1_pretrained.dat', 'wb'))
cPickle.dump(test_1, open('test_conv1_pretrained.dat', 'wb'))

np.random.seed(2)

model = Sequential()
model.add(Embedding(max_features+1, 200, input_length=X_train.shape[1], weights=[W]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1, shuffle=True)

train_2 = model.predict_proba(X_train, batch_size=128)
test_2 = model.predict_proba(X_test)

cPickle.dump(train_2, open('train_conv2_pretrained.dat', 'wb'))
cPickle.dump(test_2, open('test_conv2_pretrained.dat', 'wb'))

np.random.seed(3)

model = Sequential()
model.add(Embedding(max_features+1, 200, input_length=X_train.shape[1], weights=[W]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1, shuffle=True)

train_3 = model.predict_proba(X_train, batch_size=128)
test_3 = model.predict_proba(X_test)

cPickle.dump(train_3, open('train_conv3_pretrained.dat', 'wb'))
cPickle.dump(test_3, open('test_conv3_pretrained.dat', 'wb'))

np.random.seed(4)

model = Sequential()
model.add(Embedding(max_features+1, 200, input_length=X_train.shape[1], weights=[W]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1, shuffle=True)

train_4 = model.predict_proba(X_train, batch_size=128)
test_4 = model.predict_proba(X_test)

cPickle.dump(train_4, open('train_conv_lstm_pretrained.dat', 'wb'))
cPickle.dump(test_4, open('test_conv_lstm_pretrained.dat', 'wb'))

np.random.seed(5)

model = Sequential()
model.add(Embedding(max_features+1, 200, input_length=X_train.shape[1], weights=[W]))
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1, shuffle=True)

train_5 = model.predict_proba(X_train, batch_size=128)
test_5 = model.predict_proba(X_test)

cPickle.dump(train_5, open('train_lstm_pretrained.dat', 'wb'))
cPickle.dump(test_5, open('test_lstm_pretrained.dat', 'wb'))

dumpFeatures(full=True, n_gram=True, pretrained=False, nb_words=20000, namefile='bigram_features.dat')

[X_train, y, X_test, max_features] = cPickle.load(open("bigram_features.dat", "rb"))

np.random.seed(6)

model = Sequential()
model.add(Embedding(max_features+1, 50, input_length=X_train.shape[1]))
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1)

train_6 = model.predict_proba(X_train, batch_size=128)
test_6 = model.predict_proba(X_test)

cPickle.dump(train_6, open('train_fasttext_bigram.dat', 'wb'))
cPickle.dump(test_6, open('test_fasttext_bigram.dat', 'wb'))

np.random.seed(7)

model = Sequential()
model.add(Embedding(max_features+1, 50, input_length=X_train.shape[1]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=128, verbose=1, shuffle=True)

train_7 = model.predict_proba(X_train, batch_size=128)
test_7 = model.predict_proba(X_test)

cPickle.dump(train_7, open('train_conv1_bigram.dat', 'wb'))
cPickle.dump(test_7, open('test_conv1_bigram.dat', 'wb'))

np.random.seed(8)

model = Sequential()
model.add(Embedding(max_features+1, 20, input_length=X_train.shape[1]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=128, verbose=1, shuffle=True)

train_8 = model.predict_proba(X_train, batch_size=128)
test_8 = model.predict_proba(X_test)

cPickle.dump(train_8, open('train_conv2_bigram.dat', 'wb'))
cPickle.dump(test_8, open('test_conv2_bigram.dat', 'wb'))

np.random.seed(9)

model = Sequential()
model.add(Embedding(max_features+1, 50, input_length=X_train.shape[1]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=128, verbose=1, shuffle=True)

train_9 = model.predict_proba(X_train, batch_size=128)
test_9 = model.predict_proba(X_test)

cPickle.dump(train_9, open('train_conv_lstm_bigram.dat', 'wb'))
cPickle.dump(test_9, open('test_conv_lstm_bigram.dat', 'wb'))

np.random.seed(10)

model = Sequential()
model.add(Embedding(max_features+1, 50, input_length=X_train.shape[1]))
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=128, verbose=1, shuffle=True)

train_10 = model.predict_proba(X_train, batch_size=128)
test_10 = model.predict_proba(X_test)

cPickle.dump(train_10, open('train_lstm_bigram.dat', 'wb'))
cPickle.dump(test_10, open('test_lstm_bigram.dat', 'wb'))