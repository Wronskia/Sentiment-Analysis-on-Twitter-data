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

"""
Dumps the features

"""
dumpFeatures(full=True, n_gram=False, pretrained=True, nb_words=None, namefile='pretrained_features.dat')

"""
Load the features from the pickled ones

"""

[X_train, y, X_test, max_features, W] = cPickle.load(open("pretrained_features.dat", "rb"))

"""
Set the seed for model 1

"""
np.random.seed(1)

"""
Using keras, we define the first model with one embedding layer and
one convolutional layer followed by one maxpooling layer and 2 Dense layers
with both reLu and sigmoid activation functions

Here for model 1 to 5 we used the glove200 pretrained embedding (200 stands for the dimension of the word vectors)
weights=[W] is the argument given to the embedding W is then the matrix built using glove

Also, for all models we used binary_crossentropy as a measure of the loss and
after testing some other optimizers like adadelta we chose to fit all our models with Adam optimizer
with default learning rate of 0.001

"""
model = Sequential()
model.add(Embedding(max_features+1, 200, input_length=X_train.shape[1], weights=[W]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

"""
Fitting with 0.1 validation split

"""

model.fit(X_train, y, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1)
train_1 = model.predict_proba(X_train, batch_size=128)
test_1 = model.predict_proba(X_test)

"""
Dump the results of model 1

"""

cPickle.dump(train_1, open('features/train/train_conv1_pretrained.dat', 'wb'))
cPickle.dump(test_1, open('features/test/test_conv1_pretrained.dat', 'wb'))

"""
Set the seed for model 2

"""

np.random.seed(2)

"""
Same model as before with different seed

"""

model = Sequential()
model.add(Embedding(max_features+1, 200, input_length=X_train.shape[1], weights=[W]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

"""
Fitting with 0.1 validation split

"""

model.fit(X_train, y, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1, shuffle=True)

train_2 = model.predict_proba(X_train, batch_size=128)
test_2 = model.predict_proba(X_test)

"""
Dump the results of model 2

"""

cPickle.dump(train_2, open('features/train/train_conv2_pretrained.dat', 'wb'))
cPickle.dump(test_2, open('features/test/test_conv2_pretrained.dat', 'wb'))

"""
Set the seed for model 3

"""

np.random.seed(3)

"""
Same model as before with different seed

"""

model = Sequential()
model.add(Embedding(max_features+1, 200, input_length=X_train.shape[1], weights=[W]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

"""
Fitting with 0.1 validation split

"""

model.fit(X_train, y, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1, shuffle=True)

train_3 = model.predict_proba(X_train, batch_size=128)
test_3 = model.predict_proba(X_test)

"""
Dump the results of model 3

"""

cPickle.dump(train_3, open('features/train/train_conv3_pretrained.dat', 'wb'))
cPickle.dump(test_3, open('features/test/test_conv3_pretrained.dat', 'wb'))

"""
Set the seed for model 4

"""

np.random.seed(4)

"""
Using keras, we define the first model with one embedding layer and
one convolutional layer followed by one maxpooling layer and LSTM layer and 2 Dense layers
with both reLu and sigmoid activation functions

(CONV + LSTM + pretrained Glove200)

"""

model = Sequential()
model.add(Embedding(max_features+1, 200, input_length=X_train.shape[1], weights=[W]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

"""
Fitting with 0.1 validation split

"""

model.fit(X_train, y, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1, shuffle=True)

train_4 = model.predict_proba(X_train, batch_size=128)
test_4 = model.predict_proba(X_test)

"""
Dump the results of model 4

"""

cPickle.dump(train_4, open('features/train/train_conv_lstm_pretrained.dat', 'wb'))
cPickle.dump(test_4, open('features/test/test_conv_lstm_pretrained.dat', 'wb'))

"""
Set the seed for model 5

"""

np.random.seed(5)

"""
Model 5 : Embedding + pretraining + LSTM + Dense layer with a sigmoid activation

"""

model = Sequential()
model.add(Embedding(max_features+1, 200, input_length=X_train.shape[1], weights=[W]))
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

"""
Fitting with 0.1 validation split

"""

model.fit(X_train, y, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1, shuffle=True)

train_5 = model.predict_proba(X_train, batch_size=128)
test_5 = model.predict_proba(X_test)

"""
Dump the results of model 5

"""

cPickle.dump(train_5, open('features/train/train_lstm_pretrained.dat', 'wb'))
cPickle.dump(test_5, open('features/test/test_lstm_pretrained.dat', 'wb'))

"""
Load the features with the 2-grams and without the pretrained embeddings

"""

dumpFeatures(full=True, n_gram=True, pretrained=False, nb_words=20000, namefile='bigram_features.dat')

[X_train, y, X_test, max_features] = cPickle.load(open("bigram_features.dat", "rb"))

"""
Set the seed for model 6

"""

np.random.seed(6)

"""
Model 6 : Embedding + GlobalAveragePooling + Dense layer with a sigmoid activation

For model 6 to 10, we don't use glove pretrained features but we add the 2-grams
so we have different set of features

"""

model = Sequential()
model.add(Embedding(max_features+1, 50, input_length=X_train.shape[1]))
model.add(GlobalAveragePooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

"""
Fitting with 0.1 validation split

"""

model.fit(X_train, y, validation_split=0.1, nb_epoch=2, batch_size=128, verbose=1)

train_6 = model.predict_proba(X_train, batch_size=128)
test_6 = model.predict_proba(X_test)

"""
Dump the results of model 6

"""

cPickle.dump(train_6, open('features/train/train_fasttext_bigram.dat', 'wb'))
cPickle.dump(test_6, open('features/test/test_fasttext_bigram.dat', 'wb'))

"""
Set the seed for model 7

"""

np.random.seed(7)

"""
Model 7 : Embedding + Convolution + Dense layers with sigmoid and reLu activation

"""

model = Sequential()
model.add(Embedding(max_features+1, 50, input_length=X_train.shape[1]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

"""
Fitting with 0.1 validation split

"""

model.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=128, verbose=1, shuffle=True)

train_7 = model.predict_proba(X_train, batch_size=128)
test_7 = model.predict_proba(X_test)

"""
Dump the results of model 7

"""

cPickle.dump(train_7, open('features/train/train_conv1_bigram.dat', 'wb'))
cPickle.dump(test_7, open('features/test/test_conv1_bigram.dat', 'wb'))

"""
Set the seed for model 8

"""

np.random.seed(8)

"""
Model 8 : Embedding + Convolution + MaxPooling+Flattening+ Dense layers with sigmoid and reLu activation

"""

model = Sequential()
model.add(Embedding(max_features+1, 20, input_length=X_train.shape[1]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

"""
Fitting with 0.1 validation split

"""

model.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=128, verbose=1, shuffle=True)

train_8 = model.predict_proba(X_train, batch_size=128)
test_8 = model.predict_proba(X_test)

"""
Dump the results of model 8

"""

cPickle.dump(train_8, open('features/train/train_conv2_bigram.dat', 'wb'))
cPickle.dump(test_8, open('features/test/test_conv2_bigram.dat', 'wb'))

"""
Set the seed for model 9

"""

np.random.seed(9)

"""
Model 9: Embedding + Convolution + MaxPooling+LSTM+ Dense layer with sigmoid activation

"""

model = Sequential()
model.add(Embedding(max_features+1, 50, input_length=X_train.shape[1]))
model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

"""
Fitting with 0.1 validation split

"""
model.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=128, verbose=1, shuffle=True)

train_9 = model.predict_proba(X_train, batch_size=128)
test_9 = model.predict_proba(X_test)

"""
Dump the results of model 9

"""

cPickle.dump(train_9, open('features/train/train_conv_lstm_bigram.dat', 'wb'))
cPickle.dump(test_9, open('features/test/test_conv_lstm_bigram.dat', 'wb'))

"""
Set the seed for model 10

"""

np.random.seed(10)

"""
Model 10: Embedding + LSTM + Dense layer

"""

model = Sequential()
model.add(Embedding(max_features+1, 50, input_length=X_train.shape[1]))
model.add(LSTM(100, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

"""
Fitting with 0.1 validation split

"""

model.fit(X_train, y, validation_split=0.1, nb_epoch=1, batch_size=128, verbose=1, shuffle=True)

train_10 = model.predict_proba(X_train, batch_size=128)
test_10 = model.predict_proba(X_test)

"""
Dump the results of model 10

"""

cPickle.dump(train_10, open('features/train/train_lstm_bigram.dat', 'wb'))
cPickle.dump(test_10, open('features/test/test_lstm_bigram.dat', 'wb'))
