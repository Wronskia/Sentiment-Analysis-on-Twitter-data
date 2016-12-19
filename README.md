# Sentiment Analysis on Twitter data 

For the course "Pattern Classification and Machine Learning" at EPFL, we worked on sentiment analysis over twitter data. This project was a competition hosted by Kaggle : https://inclass.kaggle.com/c/epfml-text/
We had at our disposal two sample files for fast testing of negative and positive labels both containing 100000 tweet
Also, we had a complete dataset of 2500000 tweets (1250000 for each label)
The dataset has been labeled by the presence of  ":)" for positive tweets and ":(" for negative tweets 
you can download the dataset on https://inclass.kaggle.com/c/epfml-text/data

## Results

As a baseline, we used fastText on the small dataset.
For the final model,and over the full dataset, we used 10 neural network models over two set of features (5 models on each): 
- `Pre-trained embedding`: The first set of features used 1-grams with pretrained word embeddings (GLoVE)
- `2-gram`: The second set of features used 1,2-gram features

After building the 10 models, we fitted XGBOOST over the matrix of probabilities (10 by 2500000) which yield the final result.

In these 10 models, we mainly used LSTM,CONVOLUTIONS,MAXPOOLING layers. We mixed them by changing the seeds and the set of features.
You can see the details of the models on final/models.
Here are the results for the 10 models and the final result :

| Models       | Accuracy           | Validation Acc |
| -------------|:------------------:|:-------------------:|
| Model 1      | 0.90197            | 0.87174             |
| Model 2      | 0.90258            | 0.87027             |
| Model 3      | 0.90233            | 0.87111             |
| Model 4      | 0.90238            | 0.87175             |
| Model 5      | 0.90662            | 0.87528             |
| Model 6      | 0.90409            | 0.86494             |
| Model 7      | 0.91412            | 0.87671             |
| Model 8      | 0.90766            | 0.87558             |
| Model 9      | 0.91390            | 0.87818             |
| Model 10     | 0.90856            | 0.87767             |

After that, we applied XGBoost over the matrix of probabilities which resulted in an accuracy of 0.91967 and a validation accuracy of 0.88416.

We scored 0.88300 on kaggle
You can see the leaderboard on : https://inclass.kaggle.com/c/epfml-text/leaderboard

After tuning the hyperparameters with fastText baseline, we did use them on the neural network models. However, beside a quick tuning, we didn't optmize the hyperparameters of the neural networks because our model was sufficient enough for the competition.
Therefore, one could improve the scores by tuning the hyperparameters and/or removing/adding other models like GRU ...

## Files/Folders

Below are the details of the files and folders :

- `TFIDF-LIGHTGBM`: This ipynb file outlines our first attempt on modeling our problem
- `Final`: The folder that contains our final model explained above

In the Final folder you can find :

- `requirements.txt`: Contains the required packages to run our model (pip install requirement)
- `features.py`: Contains the details of the building of the feature matrix
- `models.py`: Contains the details of the 10 models 
- `main.py`: Load the pickled neural network models + fits the obtained results with XGBOOST + Creates the kaggle csv submission
- `preprocess.py`: Preprocesses all the tweets (Cleaning part of the tweets)
- `dico`: This folder contains the 3 normalizing dictionnaries 
- `features`: This folder contains the pickled files of the models for both the train and test set

## How to run the code

- Start by installing the packages in the Final folder :
```
pip install requirements.txt
```
- For the baseline install fastText :

```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ make
```
Then put the files `fasttext_tuning.py` and `fasttext_files_creation.py` in the folder fastText
You can then use `fasttext_tuning.py` to optimize the parameters after creating the files for the input of fastText.

- To run the final model :

We stored all the features of the 10 models in the folder features.
To run the models using the pickled features we provide :

```
$ python3 run.py 
```
This will yield our Kaggle prediction that scored 0.88300.

We used a g2.2xlarge instance on Amazon Web Service. Actually, we runned each one of the 10 models apart (for sake of memory) and we pickled the resulting features.

However, if you want to run your personalized model or run the models from the start, please run one model at a time by commenting the 9 others in the file `models.py` and dump the features.

After dumping all the features, load them and run XGBoost on the probability matrix.
