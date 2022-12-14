from timeit import default_timer

start = default_timer()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Some first exploration of the data set
# References to: https://www.kaggle.com/code/cesarcf1977/movielens-data-analysis-beginner-s-first/notebook
# Call the file

ratings = pd.read_csv("ml-latest-small/ratings.csv")

st = default_timer()

# Load the data
movies = pd.read_csv("ml-latest-small/movies.csv")
ratings = pd.read_csv("ml-latest-small/ratings.csv")

# Organise a bit and store into feather-format
movies.sort_values(by='movieId', inplace=True)
movies.reset_index(inplace=True, drop=True)
ratings.sort_values(by='movieId', inplace=True)
ratings.reset_index(inplace=True, drop=True)

# Split title and release year in separate columns in movies dataframe. Convert year to timestamp.
movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
movies.year = pd.to_datetime(movies.year, format='%Y')
movies.year = movies.year.dt.year  # As there are some NaN years, resulting type will be float (decimals)
movies.title = movies.title.str[:-7]
print(ratings.dtypes)

# Categorize movies genres properly. Working later with +20MM rows of strings proved very resource consuming
genres_unique = pd.DataFrame(movies.genres.str.split('|').tolist()).stack().unique()
genres_unique = pd.DataFrame(genres_unique, columns=['genre'])  # Format into DataFrame to store later
movies = movies.join(movies.genres.str.get_dummies().astype(bool))
movies.drop('genres', inplace=True, axis=1)
# print(genres_unique)

# Modify rating timestamp format (from seconds to datetime year)
ratings.timestamp = pd.to_datetime(ratings.timestamp, unit='s')
ratings.timestamp = pd.to_datetime(ratings.timestamp, infer_datetime_format=True)
ratings.timestamp = ratings.timestamp.dt.year
print(ratings.head(10))

# Check and clean NaN values
print(f"Number of movies Null values: {max(movies.isnull().sum())}")
print("Number of ratings Null values: ", max(ratings.isnull().sum()))
movies.dropna(inplace=True)
ratings.dropna(inplace=True)

# Organise a bit, then save into feather-formatand clear from memory
movies.sort_values(by='movieId', inplace=True)
ratings.sort_values(by='movieId', inplace=True)
movies.reset_index(inplace=True, drop=True)
ratings.reset_index(inplace=True, drop=True)

runtime = default_timer() - st
print("Elapsed time(sec): ", round(runtime, 2))

# Explore data with some basic plots:

# 1: Number of movies and ratings per year

# Let's work with a temp smaller slice 'dftmp' of the original dataframe to reduce runtime (ratings hass +2MM rows)
st = default_timer()
dftmp = movies[['movieId', 'year']].groupby('year')
print((dftmp.head(10)))

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(dftmp.year.first(), dftmp.movieId.nunique(), "g-o")
ax1.grid(None)
ax1.set_ylim(0, )

dftmp = ratings[['rating', 'timestamp']].groupby('timestamp')
ax2 = ax1.twinx()
ax2.plot(dftmp.timestamp.first(), dftmp.rating.count(), "r-o")
ax2.grid(None)
ax2.set_ylim(0, )

ax1.set_xlabel('Year')
ax1.set_ylabel('Number of movies released');
ax2.set_ylabel('Number of ratings')
plt.title('Movies per year')
# Add a footnote below and to the right side of the chart
plt.figtext(0.5, 0.02, "Number of movies released per year increasing almost exponentially until 2009, "
                       "then flattening and dropping significantly in 2014"
            , ha="center", fontsize=9, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
plt.show()

# 2: Cumulative number of movies, in total and per genre
# INSIGHT: #2**: On average, movies are categorized into 2 genres (i.e. number of movies-genres 54k doubles the number
# of movies 27k). Comedy 8.3k and Drama 13.3k are the top genres used.

plt.figure(figsize=(10, 5))
dftmp = movies[['movieId', 'year']].groupby('year')
df = pd.DataFrame({'All_movies': dftmp.movieId.nunique().cumsum()})

# Plot histogram for each individual genre

for genre in genres_unique.genre:
    dftmp = movies[movies[genre]][['movieId', 'year']].groupby('year')
    df[genre] = dftmp.movieId.nunique().cumsum()
df.fillna(method='ffill', inplace=True)
print('df[genre] = ', df[genre])

df.loc[:, df.columns != 'All_movies'].plot.area(stacked=True, figsize=(10, 5))

# Plot histogram for all movies
plt.plot(df['All_movies'], marker='o', markerfacecolor='black')
plt.xlabel('Year')
plt.ylabel('Cumulative number of movies-genre')
plt.title('Total movies-genre')  # Many movies have multiple genres, so count here is higher than number of movies
plt.legend(loc=(1.05, 0), ncol=2)

# Plot simple scatter of the number of movies tagged with each genre
plt.figure(figsize=(15, 5))
barlist = df.iloc[-1].plot.bar()
barlist.patches[0].set_color('b')  # Color 'All_movies' differently, as it's not a genre tag count
plt.xticks(rotation=20)  # previously vertical
plt.title('Movies per genre tag')
plt.xlabel('Genre')
plt.ylabel('Number of movies tagged')
plt.show()

# 3: Distributions by genre, on top of total rating distribution.
# This will help identifying consistent ratings or outlines

st = default_timer()

dftmp = ratings[['movieId', 'rating']].groupby('movieId').mean()

# Initialize empty list to capture basic stats by gere
rating_stats = []
# Plot general histogram of all ratings
dftmp.hist(bins=25, grid=False, edgecolor='b', density=True, label='All genres', figsize=(10, 5))
# Plot histograms (kde lines for better visibility) per genre
for genre in genres_unique.genre:
    dftmp = movies[movies[genre] == True]
    dftmp = ratings[ratings.set_index('movieId').index.isin(dftmp.set_index('movieId').index)]
    dftmp = dftmp[['movieId', 'rating']].groupby('movieId').mean()
    dftmp.rating.plot(grid=False, alpha=0.6, kind='kde', label=genre)
    avg = dftmp.rating.mean()
    std = dftmp.rating.std()
    rating_stats.append((genre, avg, std))
plt.legend(loc=(1.05, 0), ncol=2)
plt.xlim(0, 5)
plt.xlabel('Movie rating')
plt.title('Movie rating histograms')
plt.show()

# 4: Compute basic statistics (avg, std) per genre. Plot dispersion (box plot).

# 4.INSIGHT: As highlighted in the previous plot, all genres really show similar behaviour.
# Using numbers in this case, mean rating is slightly above 3, with a standard deviation around 0.65.

st = default_timer()

# Construct a DataFrame with all the basic stats per genre
rating_sum = pd.DataFrame(rating_stats, columns=['genre', 'average', 'std_dev'])
rating_sum.set_index('genre', inplace=True)
# Display a boxplot to show the dispersion of average and standard deviation
bxplot = plt.boxplot(rating_sum.T, labels=['Average', 'STD'], sym="ro")
# Properly label the outliers
outliers = []
for i, value in enumerate(bxplot['fliers']):
    for val in value.get_ydata():
        bxplot_label = rating_sum.index[rating_sum.iloc[:, i] == val][0]
        outliers.append(bxplot_label)
        plt.annotate(bxplot_label, xy=(i + 1.1, val + np.random.uniform(-0.25, 0.25)))  # Trick to prevent overlapping
outliers = set(outliers)
plt.ylim(0, )
plt.ylabel('Movie rating')
plt.title('Movie rating descriptive stats')
plt.show()

# Bar chart with average rating, standard deviation and normalized number of ratings per genre
# Calculate the normalized number of ratings per genre
rating_sum['num_ratings_norm'] = df.iloc[-1, 1:] / df.iloc[-1, 1:].sum()
# Calculate the average rating for all genres
rating_sum['average_all'] = rating_sum.average.mean()

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
ax2 = ax.twinx()

rating_sum[['average', 'std_dev']].plot(kind='bar', color=['b', 'r'], ax=ax, position=0.5, grid=False)
rating_sum['average_all'].plot(kind='line', style='--', color=['black'], ax=ax, grid=False)
rating_sum['num_ratings_norm'].plot(kind='line', color='g', ax=ax2, grid=False, linewidth=3)

ax.legend(loc=2)
ax2.legend(loc=1)

ax.set_xticklabels(ax.get_xticklabels(), rotation=20)  # previously 90
plt.title('Movie rating descriptive stats')
plt.autoscale()
ax2.set_ylim(0, )
plt.figtext(0.5, 0.02, "Using numbers in this case, mean rating is slightly above 3, with a standard deviation around "
                       "0.65 "
            , ha="center", fontsize=9, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
plt.show()

print("Outliers: ", outliers)
print(rating_sum.T)

# Quick pie chart to visualize how 3 genres take almost 50% of ratings
rating_sum.sort_values(by='num_ratings_norm', inplace=True)
plt.pie(rating_sum['num_ratings_norm'], labels=rating_sum.T.columns, labeldistance=2)
plt.show()
