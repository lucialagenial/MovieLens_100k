import pandas as pd



# Visualizations from 3. on

import matplotlib
import matplotlib.pyplot as plt
#matplotlib.style.use(matplotlib.get_data_path() + '\stylelib\APA.mplstyle')  # selecting
# the style sheet

# Collaborative filtering

# Memory based

# The first category includes algorithms that are memory based, in which statistical
# techniques are applied to the entire dataset to calculate the predictions.
# To find the rating R that a user U would give to an item I, the approach includes:
# - Finding users similar to U who have rated the item I
# - Calculating the rating R based the ratings of users found in the previous step


# Call the file

ratings = pd.read_csv("ml-latest-small/ratings.csv")
print(ratings.describe())


# How to Find Similar Users on the Basis of Ratings?

