import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')

from timeit import default_timer

start = default_timer()

from scipy.sparse.linalg import svds

# Support Vector Decomposition (SVD)

# A recommendation technique that is efficient when the number of dataset is limited may be unable to generate
# satisfactory number of recommendations when the volume of dataset is increased.
# Thus, it is crucial to apply recommendation techniques which are capable of scaling up in a successful manner as the
# number of dataset in a database increases.
# Methods used for solving scalability problem and speeding up recommendation generation are based on Dimensionality
# reduction techniques, such as Singular Value Decomposition (SVD) method, which has the ability to produce reliable and
# efficient recommendations.

# Load the data
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")
tags = pd.read_csv("ml-latest-small/tags.csv")

# Modify rating timestamp format (from seconds to datetime year)
movies['release_year'] = movies['title'].str.extract(r'(?:\((\d{4})\))?\s*$', expand=False)

st = default_timer()
ratings.timestamp = pd.to_datetime(ratings.timestamp, unit='s')
ratings.timestamp = pd.to_datetime(ratings.timestamp, infer_datetime_format=True)
ratings.timestamp = ratings.timestamp.dt.year

movie_data = ratings.merge(movies, on='movieId', how='left')
movie_data = movie_data.merge(tags, on=['userId', 'movieId', 'timestamp'], how='left')



# Calculate SVD by manual
n_users = movie_data['userId'].nunique()
n_movies = movie_data['movieId'].nunique()

print('Number of users:', n_users)
print('Number of movies:', n_movies)

final_df_matrix = movie_data.pivot(index='userId',
                                   columns='movieId',
                                   values='rating').fillna(0)

print(final_df_matrix.head())

user_ratings_mean = np.mean(final_df_matrix.values, axis=1)
ratings_demeaned = final_df_matrix.values - user_ratings_mean.reshape(-1, 1)

# Check sparsity
sparsity = round(1.0 - movie_data.shape[0] / float(n_users * n_movies), 3)
print('The sparsity level of MovieLens100k dataset is ' + str(sparsity * 100) + '%')

U, sigma, Vt = svds(ratings_demeaned, k=50)  # Number of singular values and vectors to compute.

# To leverage matrix multiplication to get predictions, I'll convert the  Σ  (now are values) to the diagonal matrix
# form.

sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

preds = pd.DataFrame(all_user_predicted_ratings, columns=final_df_matrix.columns)
print(preds.head())


# Function to return movies with the highest predicted ration that the specified user hasn't already rated.
# with no explicit movie content feature (such as genre or title).

print(movie_data.columns)

def recommend_movies(predictions, userID, movies, ratings, num_recommendations):
    # Get and sort the user's predictions
    user_row_number = userID - 1  # User ID starts at 1, not 0
    sorted_user_predictions = preds.iloc[user_row_number].sort_values(ascending=False)

    # Get the user's data and merge in the movie information.
    user_data = ratings[ratings.userId == userID]
    user_full = (user_data.merge(movies, how='left', on='movieId').
                 sort_values(['rating'], ascending=False)
                 )

    print('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print('Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies[~movies['movieId'].isin(user_full['movieId'])].
                           merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                                 left_on='movieId',
                                 right_on='movieId').
                           rename(columns={user_row_number: 'Predictions'}).
                           sort_values('Predictions', ascending=False).
                           iloc[:num_recommendations, :-1])

    return user_full.head(10), recommendations.sort_values('release_year', ascending=False) # then sort by newest release year


user_already_rated, for_recommend = recommend_movies(preds, 2, movies, ratings, 10)
print(user_already_rated)

# Source: https://www.kaggle.com/code/indralin/movielens-project-1-2-collaborative-filtering/notebook