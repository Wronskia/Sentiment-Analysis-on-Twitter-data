import nltk
import itertools
from nltk.tokenize import TweetTokenizer
import pandas as pd
import numpy as np
import re
import enchant
from nltk.corpus import stopwords


'''
arrangeTweet Takes a tweet as Input and normalize each word
for example : 'llooooooovvvee' becomes 'love'
'''

d = enchant.Dict("en_US")
def arrangeTweet(tweet):
    tweet=tweet.lower()
    tweet=tweet.split()
    for i in range(len(tweet)):
        if d.check(''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet[i]))):
            tweet[i]=''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet[i]))
        else:
            tweet[i]=''.join(''.join(s)[:1] for _, s in itertools.groupby(tweet[i]))
    tweet=' '.join(tweet)
    return tweet
    
    
'''
Cleaning of the tweets using regular expressions and NLTK for
removing the stop words.
'''

stopwordlist = set(stopwords.words("english"))
def clean(tweet):
    tweet = re.sub(r"\'s", " \'s", tweet)
    tweet = re.sub(r"\'ve", " \'ve", tweet)
    tweet = re.sub(r"n\'t", " n\'t", tweet)
    tweet = re.sub(r"\'re", " \'re", tweet)
    tweet = re.sub(r"\'d", " \'d", tweet)
    tweet = re.sub(r"\'ll", " \'ll", tweet)
    tweet = re.sub(r",", " , ", tweet)
    tweet = re.sub(r"!", " ! ", tweet)
    tweet = re.sub(r"\(", " \( ", tweet)
    tweet = re.sub(r"\)", " \) ", tweet)
    tweet = re.sub(r"\?", " \? ", tweet)
    tweet = re.sub(r"\s{2,}", " ", tweet)
    tweet = arrangeTweet(tweet) 
    liste=[word for word in tweet.split() if word not in stopwordlist]
    liste=[word for word in liste if len(word)>1]
    tweet=' '.join(liste)
    return tweet.strip().lower()
    
    
'''
Building the training and validation files for the fastText Input
FastTextInput() creates both files : fastText_training.txt and fastText_validation.txt
which will be what we will feed the algorithm
This will be our baseline :
'''


def FastTextInput():
    i=0
    IDs=[]
    posfile=open("train_pos.txt",'rb')
    negfile=open("train_neg.txt",'rb')
    trainfile=open("fastText_training.txt",'w')
    validationfile=open("fastText_validation.txt",'w')
    for line in posfile:
        if i>90000:
            if i%10000==0:
                print(i)
            i+=1
            line = line.decode('utf8')
            line=clean(line)  
            validationfile.write("__label__1 ,"+line+"\n")
        else:
            if i%10000==0:
                print(i)
            i+=1
            line = line.decode('utf8')
            line=clean(line)  
            trainfile.write("__label__1 ,"+line+"\n")
    i=0
    for line in negfile:
        if i>90000:
            if i%100000==0:
                print(i)
            i+=1
            line = line.decode('utf8')
            line=clean(line)  
            validationfile.write("__label__2 ,"+line+"\n")
        else:
            if i%100000==0:
                print(i)
            i+=1
            line = line.decode('utf8')
            line=clean(line)  
            trainfile.write("__label__2 ,"+line+"\n")

    return
    
    
    
FastTextInput()
