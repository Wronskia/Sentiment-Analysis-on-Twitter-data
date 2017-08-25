import sys
from preprocess import clean
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import _pickle as cPickle

def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list)-ngram_range+1):
            for ngram_value in range(2, ngram_range+1):
                ngram = tuple(new_list[i:i+ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences



def train_test_features(full=True, n_gram=False, pretrained=True, nb_words=None):
"""
Function that does the cleaning,the tokenization, add the n-grams, build the weight matrix(pretrained)
depending on the arguments
Arguments: full (True to use the full dataset)
           n_gram (True to use 2-grams and False to use 1-grams) we didn't use more than 2-grams
           because of the limit of 140 characters in twitter
           pretrained (True to use the glove200 pretraining)
           nb_words (None to take all the words otherwise takes an integer value N which will take the N most frequent words)
Returns : train_sequences (List of list of word indexes for each tweet for the training)
          test_sequences (List of list of word indexes for each tweet for the test)
          labels (Labels)
          max_features (Maximum word index in the list of list train_sequences)
          embedding_matrix (if pretrained=True returns the embedding_matrix builded by the help of glove)

          This code was inspired from : https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py
          and from : https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py

"""
    np.random.seed(0)

    if full:
        posfile = open("train_pos_full.txt", 'rb')
        negfile = open("train_neg_full.txt", 'rb')
        train_size = 2500000
    else:
        posfile = open("train_pos.txt", 'rb')
        negfile = open("train_neg.txt", 'rb')
        train_size = 200000

    train_tweets = []
    count = 0

    print('Processing of' + str(train_size) + 'tweets for the training set...')
    for tweet in posfile:
        tweet = tweet.decode('utf8')
        tweet = clean(tweet)
        train_tweets.append(tweet)
        count = count + 1
        if count%1000 == 0:
            sys.stdout.write('\r{0}'.format(count) + '/' + str(train_size) + ' processed tweets.')
            sys.stdout.flush()
    posfile.close()

    for tweet in negfile:
        tweet = tweet.decode('utf8')
        tweet = clean(tweet)
        train_tweets.append(tweet)
        count = count + 1
        if count%1000 == 0:
            sys.stdout.write('\r{0}'.format(count) + '/' + str(train_size) + ' processed tweets.')
            sys.stdout.flush()
    negfile.close()

    test_size = 10000
    test_tweets = []
    testfile = open('test_data.txt' ,'rb')
    count = 0
    print('\nProcessing of' + str(test_size) + 'tweets for the testing set...')
    for tweet in testfile:
        tweet = tweet.decode('utf8')
        tweet = tweet[tweet.find(',')+1:]
        tweet = clean(tweet)
        test_tweets.append(tweet)
        count = count + 1
        if count%1000 == 0:
            sys.stdout.write('\r{0}'.format(count) + '/' + str(test_size) + ' processed tweets.')
            sys.stdout.flush()

    print('\nTokenization of tweets...')
    if nb_words == None:
        tokenizer = Tokenizer(filters='')
    else:
        tokenizer = Tokenizer(nb_words=nb_words, filters='')
    tokenizer.fit_on_texts(train_tweets)
    word_index = tokenizer.word_index
    max_features = len(word_index)
    print('Found %s unique tokens.' % max_features)

    train_sequences = tokenizer.texts_to_sequences(train_tweets)
    test_sequences = tokenizer.texts_to_sequences(test_tweets)

    maxlen = 30

    if n_gram:
        maxlen = 60
        ngram_range = 2
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in train_sequences:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        max_features = np.max(list(indice_token.keys())) + 1

        # Augmenting X_train and X_test with n-grams features
        train_sequences = add_ngram(train_sequences, token_indice, ngram_range)
        test_sequences = add_ngram(test_sequences, token_indice, ngram_range)
        print('Average train sequence length: {}'.format(np.mean(list(map(len, train_sequences)), dtype=int)))
        print('Average test sequence length: {}'.format(np.mean(list(map(len, test_sequences)), dtype=int)))

    print('Pad sequences (samples x time)')
    train_sequences = sequence.pad_sequences(train_sequences, maxlen=maxlen)
    test_sequences = sequence.pad_sequences(test_sequences, maxlen=maxlen)
    print('train_sequences shape:', train_sequences.shape)
    print('test_sequences shape:', test_sequences.shape)

    labels = np.array(int(train_size/2) * [0] + int(train_size/2) * [1])

    print('Shuffling of the training set...')
    indices = np.arange(train_sequences.shape[0])
    np.random.shuffle(indices)
    train_sequences = train_sequences[indices]
    labels = labels[indices]

    if pretrained:

        print('Extracting word vectors from glove200.txt...')
        embeddings_index = {}
        f = open('embeddings/glove200.txt','rb')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))

        print('Creating the embedding matrix...')
        embedding_matrix = np.zeros((max_features + 1, 200))
        for word, i in word_index.items():
            if i > max_features:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        print('Embedding matrix created.')
        return train_sequences, labels, test_sequences, max_features, embedding_matrix
    else:
        return train_sequences, labels, test_sequences, max_features

def dumpFeatures(full, n_gram, pretrained, nb_words, namefile):
    """
    Function that dumps the output of the function above : train_test_features()
    Arguments: full (True to use the full dataset)
               n_gram (True to use 2-grams and False to use 1-grams) we didn't use more than 2-grams
               because of the limit of 140 characters in twitter
               pretrained (True to use the glove200 pretraining)
               namefile (the name of the file that will contain the output)
               nb_words (None to take all the words otherwise takes an integer value N which will take the N most frequent words)

    """
    if pretrained:
        train_sequences, labels, test_sequences, max_features, embedding_matrix = train_test_features(full, n_gram, pretrained, nb_words)
        cPickle.dump([train_sequences, labels, test_sequences, max_features, embedding_matrix],open(namefile, 'wb'))
    else:
        train_sequences, labels, test_sequences, max_features = train_test_features(full, n_gram, pretrained, nb_words)
        cPickle.dump([train_sequences, labels, test_sequences, max_features], open(namefile, 'wb'))
    return
