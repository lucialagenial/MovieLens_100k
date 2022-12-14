# MovieLens_100k
*"Recommendations systems will take us from the age of information to the age of recommendation."*

This is a project for the **Data Circle Course** in the **ReDI School** for Digital Integration Berlin 2022 winter term.

We, *Lucía Morales Lizárraga* and *Chales Emeka Onyi*, use the Data set from MovieLens 100k to try different approaches, 
to get an understanding of the logic, techniques and methods behind recommendation systems. 
We attempt to answer the following questions:
1) How do you determine which users or items are similar to one another?
2) Given that you know how do you know which users are similar, how do you determine the rating that a user would give
to an item based on the rating of similar users?
3) How do you measure the accuracy of the ratings you calculate?


In general, there is mainly two types of recommender system:

###1) **Content-based**
This recommendation is based on a similar feature of different entities.
E.g. If someone likes the movie Iron man, it will recommend The Avengers, for instance, because both have similar 
   genres, actors and/or storylines. Recommender systems can extract similar features from a different entity 
   based on featured actor, genres, music, director.

###2) **Collaborative filtering**
This approach recommends the users according to the preference of other users.
There are two different methods of collaborative filtering:

a. Model-based

b. Memory-based

####2.1) Model-based:

A model-based collaborative filtering recommendation system uses a model to predict that the user will like the
recommendation or not using previous data as a dataset.

####2.2) Memory-based
In memory-based collaborative filtering recommendation based on its previous data of preference of users and recommend
that to other users.

## Approaches taken: 

We have tried the following approaches: 
- Data analysis and exploration. 

- Content based correlation recommendation system. See file: correlation_recommendation.py.
  
- Model based collaborative filtering with Singular Value Decomposition (SVD) method. See file: model_recommender_SVD.py.
  
- Model based collaborative filtering with Singular Value Decomposition (SVD) calculated manually. See file: manual_SVD.py.





###Data:

Data set from MovieLens can be found here: https://grouplens.org/datasets/movielens/
Data Source: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872 


###More info about the ReDI School: 
https://www.redi-school.org/


