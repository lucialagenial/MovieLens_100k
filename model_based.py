import pandas as pd
import numpy as np
from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans
#from load_data import data
from recommenders import algo


# Load the data
# movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")


# We are using the ratings.csv
# Modify rating timestamp format (from seconds to datetime year)
ratings.timestamp = pd.to_datetime(ratings.timestamp, unit='s')
ratings.timestamp = pd.to_datetime(ratings.timestamp, infer_datetime_format=True)
ratings.timestamp = ratings.timestamp.dt.year
#print(ratings.head(50))

# specify the rating scale. The default is (1, 5)
reader = Reader(rating_scale=(1, 5))


## Algorithms Based on K-Nearest Neighbours (k-NN)
# To use item-based cosine similarity
sim_options = {
    "name": "cosine",
    "user_based": False,  # Compute  similarities between items
}
algo = KNNWithMeans(sim_options=sim_options)

print(f"Number of users: {ratings['userId'].nunique()}")

#  to find out how the user E would rate the movie 2:

trainingSet = ratings.build_full_trainset()

algo.fit(trainingSet)

pip install --upgrade setuptools lightfm

pip install --upgrade setuptools wheel