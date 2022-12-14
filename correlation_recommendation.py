import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

from surprise import Reader, Dataset, SVD, SVDpp
from surprise import accuracy

# Content based correlation recommendation system.

# Load the data
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")
tags = pd.read_csv("ml-latest-small/tags.csv")


movie_data = ratings.merge(movies, on='movieId', how='left')
movie_data = movie_data.merge(tags, on=['userId', 'movieId', 'timestamp'], how='left')
print(movie_data.head(10))

print(movie_data.columns)

rating = pd.DataFrame(movie_data.groupby('title')['rating'].mean())
print(rating.head(10))

# Total number of ratings given to each number.

rating['Total Rating']=pd.DataFrame(movie_data.groupby('title')['rating'].count())
print(f"Total number of ratings given to each number: {rating.head(10)}")


# Correlation between users and rating of a particular movie

movie_user = movie_data.pivot_table(index='userId',columns='title',values='rating')
print(f"Correlation between users and rating of a particular movie: {movie_user.head(10)}")
# It's ok if we just see NaNs

# Check the recommender system with any movie. E.g. Iron Man (2008)
correlation = movie_user.corrwith(movie_user['Iron Man (2008)'])
correlation.head(10)
print(f"Check the recommender system with any movie. E.g. Iron Man (2008) {correlation.head(10)}")

# Remove all empty values
recommendation = pd.DataFrame(correlation,columns=['correlation'])
recommendation.dropna(inplace=True)
recommendation = recommendation.join(rating['Total Rating'])
print(f"Data without nan's: {recommendation.head()}")

# Selecting the movies that has at least 100 ratings.
recc = recommendation[recommendation['Total Rating'] > 100].sort_values('correlation',ascending=False).reset_index()
recc = recc.merge(movies, on='title', how='left')
print(f"Movies with more than 100 ratings: {recc.count()}")

print(f"Top 5 recommended movies: {recc.head(5)}")

# We learn that:
# 1. What is a recommender system?
# 2. Implementation of recommended system in python.

# Source: https://www.codespeedy.com/build-recommender-systems-with-movielens-dataset-in-python/