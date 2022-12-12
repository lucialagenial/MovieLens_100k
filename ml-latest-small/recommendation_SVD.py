import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

import warnings

warnings.filterwarnings('ignore')

from surprise import Reader, Dataset, SVD, SVDpp
from surprise import accuracy

from timeit import default_timer

start = default_timer()

# Load the data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
tags = pd.read_csv("tags.csv")


# Modify rating timestamp format (from seconds to datetime year)
st = default_timer()
ratings.timestamp = pd.to_datetime(ratings.timestamp, unit='s')
ratings.timestamp = pd.to_datetime(ratings.timestamp, infer_datetime_format=True)
ratings.timestamp = ratings.timestamp.dt.year

movie_data = ratings.merge(movies, on='movieId', how='left')
movie_data = movie_data.merge(tags, on=['userId', 'movieId', 'timestamp'], how='left')

print(movie_data.columns)

print(movie_data['timestamp'].max())

reader = Reader(rating_scale=(1, 5))

dataset = Dataset.load_from_df(movie_data[['userId', 'movieId', 'rating']], reader=reader)

svd = SVD(n_factors=50)
svd_plusplus = SVDpp(n_factors=50)

trainset = dataset.build_full_trainset()

svd.fit(trainset)  # old version use svd.train

### It will take a LONG....TIME...., but it'll give a better score in RMSE & MAE

# svd_plusplus.fit(trainset)

id_2_names = dict()

for idx, names in zip(movies['movieId'], movies['title']):
    id_2_names[idx] = names


# Function that eliminates all the movies that are not rated

def Build_Anti_Testset4User(user_id):
    fill = trainset.global_mean
    anti_testset = list()
    u = trainset.to_inner_uid(user_id)

    # ur == users ratings
    user_items = set([item_inner_id for (item_inner_id, rating) in trainset.ur[u]])

    anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                     i in trainset.all_items() if i not in user_items]

    return anti_testset


# First, let's try SVD for make Top-N recommendationFirst, let's try SVD for make Top-N recommendation


def TopNRecs_SVD(user_id, num_recommender=10, latest=False):
    testSet = Build_Anti_Testset4User(user_id)
    predict = svd.test(testSet)  # we can change to SVD++ later

    recommendation = list()

    for userID, movieID, actualRating, estimatedRating, _ in predict:
        intMovieID = int(movieID)
        recommendation.append((intMovieID, estimatedRating))

    recommendation.sort(key=lambda x: x[1], reverse=True)

    movie_names = []
    movie_ratings = []

    for name, ratings in recommendation[:20]:
        movie_names.append(id_2_names[name])
        movie_ratings.append(ratings)

    movie_dataframe = pd.DataFrame({'title': movie_names,
                                    'rating': movie_ratings}).merge(movie_data[['title', 'timestamp']],
                                                                    on='title', how='left')

    if latest == True:
        return movie_dataframe.sort_values('timestamp', ascending=False)[['title', 'rating']].head(
            num_recommender)

    else:
        return movie_dataframe.drop('timestamp', axis=1).head(num_recommender)


user_2 = Build_Anti_Testset4User(2)



print(TopNRecs_SVD(1920, num_recommender=10))
