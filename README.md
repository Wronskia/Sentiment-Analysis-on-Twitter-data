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

After that, we applied XGBOOST over the matrix of probabilities which resulted in an accuracy of $0.91967$ and a validation accuracy of $0.88416$.

We scored $0.88300$ on kaggle
You can see the leaderboard on : https://inclass.kaggle.com/c/epfml-text/leaderboard

After tuning the hyperparameters with fastText baseline, we did use them on the neural network models. However, beside a quick tuning, we didn't optmize the hyperparameters of the neural networks because our model was sufficient enough for the competition.
Therefore, one could improve the scores by tuning the hyperparameters and/or removing/adding other models like GRU ...

## Folders


