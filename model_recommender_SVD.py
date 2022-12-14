import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from surprise import Reader, Dataset, SVD, SVDpp
from timeit import default_timer
start = default_timer()

# Singular Value Decomposition (SVD) method, model based

# Load the data
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")
tags = pd.read_csv("ml-latest-small/tags.csv")

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

    recommendation.sort(key=lambda x: x[0], reverse=True)  # here is the issue x[1] originally
    # For further info trying to fix this error:
    # https://stackoverflow.com/questions/8966538/syntax-behind-sortedkey-lambda/
    # 42966511#42966511?newreg=efc951fd9eec4b3da42fa8d5a6afdc98
    movie_names = []
    movie_ratings = []

    for name, ratings in recommendation[:20]:
        movie_names.append(id_2_names[name])
        movie_ratings.append(ratings)

    movie_dataframe = pd.DataFrame({'title': movie_names,
                                    'rating': movie_ratings}).merge(movie_data[['title', 'timestamp']],
                                                                    on='title', how='left')

    if latest:
        return movie_dataframe.sort_values('timestamp', ascending=False)[['title', 'rating']].head(
            num_recommender)

    else:
        return movie_dataframe.drop('timestamp', axis=1).head(num_recommender)


user_2 = Build_Anti_Testset4User(2)

print(TopNRecs_SVD(2, num_recommender=10))
# Somehow it's working but not really

# Model evaluation

# Than predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()

predictions_svd = svd.test(testset)


# These next two lines are commented out due to the time it takes to rerun everything

#print('SVD - RMSE:', accuracy.rmse(predictions_svd, verbose=False))
#print('SVD - MAE:', accuracy.mae(predictions_svd, verbose=False))
# Remember in recommendation, the most important is Top-N recommendation (list of product to recommend), not RMSE or MAE


# Function to give a recommendation to all users

from collections import defaultdict

def GetTopN(predictions, n=10, minimumRating=4.0):
        topN = defaultdict(list)

        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            if (estimatedRating >= minimumRating):
                topN[int(userID)].append((int(movieID), estimatedRating))

        for userID, ratings in topN.items():
            ratings.sort(key=lambda x: x[1], reverse=True) # here the same issue
            topN[int(userID)] = ratings[:n]

        return topN


top_n = GetTopN(predictions_svd, n=10)

ii = 0
for uid, predict_ratings in top_n.items():
    print(uid, [iid for (iid, _) in predict_ratings])
    ii += 1

    if ii > 5:
        break


# Recommendation System take us out from the age of information and bring us in to the age of recommendation


# References: https://www.kaggle.com/code/indralin/movielens-project-1-2-collaborative-filtering/
# notebook#Support-Vector-Decomposition-(SVD)