import pandas as pd
import numpy as np


# Movie Recommendation based on K-Nearest-Neighbors (KNN)

# We'll try to guess the rating of a movie by looking at the 10 movies that are closest to it in terms of genres and
# popularity.

# Load the data
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")
tags = pd.read_csv("ml-latest-small/tags.csv")

# Group the movie ID, and compute the total number of ratings and the average for every movie.
movieProperties = ratings.groupby('movieId').agg({'rating': [np.size, np.mean]})
print(movieProperties.head())

# New df that contains the normalized number of ratings. So, a value of 0 means nobody rated it, and a value of 1 will
# mean it's the most popular movie there is.

movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print(movieNormalizedNumRatings.head())

# Add the genre information:
# The way this works is there are 19 fields, each corresponding to a specific genre - a value of '0' means it is not in
# that genre, and '1' means it is in that genre. A movie may have more than one genre associated with it.

# While we're at it, we'll put together everything into one big Python dictionary called movieDict. Each entry will
# contain the movie name, list of genre values, the normalized popularity score, and the average rating for each movie:

movieDict = {}

#with open(r'ml-latest-small/movies.csv', encoding="ISO-8859-1") as f:
temp = ''
for line in movies:
    line.encode().decode("ISO-8859-1")
    fields = line.rstrip('\n').split('|')
    print(fields)
    movieID = fields[0]
    name = fields[1]
    genres = fields[2]
    genres = map(int, genres)
    movieDict[movieID] = (name, np.array(list(genres)), movieNormalizedNumRatings.loc[movieID].get('size'),
                          movieProperties.loc[movieID].rating.get('mean'))




# For example, here's the record we end up with for movie ID 1, "Toy Story":)

#print(movieDict[1])
