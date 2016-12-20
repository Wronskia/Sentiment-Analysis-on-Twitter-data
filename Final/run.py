import numpy as np
import csv
import _pickle as cPickle
import xgboost as xgb



def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

"""
Loading the pickled files of the models

"""

train1 = cPickle.load(open("features/train/train_conv1_pretrained.dat", "rb"))
train2 = cPickle.load(open("features/train/train_conv2_pretrained.dat", "rb"))
train3 = cPickle.load(open("features/train/train_conv3_pretrained.dat", "rb"))
train4 = cPickle.load(open("features/train/train_conv_lstm_pretrained.dat", "rb"))
train5 = cPickle.load(open("features/train/train_lstm_pretrained.dat", "rb"))
train6 = cPickle.load(open("features/train/train_fasttext_bigram.dat", "rb"))
train7 = cPickle.load(open("features/train/train_conv1_bigram.dat", "rb"))
train8 = cPickle.load(open("features/train/train_conv2_bigram.dat", "rb"))
train9 = cPickle.load(open("features/train/train_conv_lstm_bigram.dat", "rb"))
train10 = cPickle.load(open("features/train/train_lstm_bigram.dat", "rb"))

test1 = cPickle.load(open("features/test/test_conv1_pretrained.dat", "rb"))
test2 = cPickle.load(open("features/test/test_conv2_pretrained.dat", "rb"))
test3 = cPickle.load(open("features/test/test_conv3_pretrained.dat", "rb"))
test4 = cPickle.load(open("features/test/test_conv_lstm_pretrained.dat", "rb"))
test5 = cPickle.load(open("features/test/test_lstm_pretrained.dat", "rb"))
test6 = cPickle.load(open("features/test/test_fasttext_bigram.dat", "rb"))
test7 = cPickle.load(open("features/test/test_conv1_bigram.dat", "rb"))
test8 = cPickle.load(open("features/test/test_conv2_bigram.dat", "rb"))
test9 = cPickle.load(open("features/test/test_conv_lstm_bigram.dat", "rb"))
test10 = cPickle.load(open("features/test/test_lstm_bigram.dat", "rb"))


"""
Building from the pickled files the probability matrix for the train and the test set

"""
train = np.hstack((train1, train2, train3, train4, train5, train6, train7, train8, train9, train10))
test = np.hstack((test1, test2, test3, test4, test5, test6, test7, test8, test9, test10))

y = np.array(1250000 * [0] + 1250000 * [1])
np.random.seed(0)
np.random.shuffle(y)

"""
Fitting XGBOOST over the probability matrix obtained from our models

"""

model = xgb.XGBClassifier().fit(train, y)

y_pred = model.predict(test)
y_pred = 1 - 2*y_pred
create_csv_submission(np.arange(1,10001),y_pred,'submission.csv')
