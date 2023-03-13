#!/usr/bin/env python
# coding: utf-8

# # Recommender systems - practical exercise 1
# non-personalized recommendations

# ## program

# In[252]:


import pandas as pd
from functools import lru_cache


# In[270]:


def read_dat_files():
    R = pd.read_csv("/ml-1m/ratings.dat",sep="::",engine="python",names=["userID","movieID","rating","timestamp"])
    del R["timestamp"]
    I = pd.read_csv("/ml-1m/movies.dat",sep="::",engine="python",names=["movieID","title","genre"])
    U = pd.read_csv("/ml-1m/users.dat",sep="::",engine="python")
    return R, I, U 


# In[271]:


R, I, U = read_dat_files()
movies


# In[255]:


@lru_cache
def calculate_simple_association(movie_id_1,movie_id_2):
    movie_1 = R[R.movieID==movie_id_1] # X
    if len(movie_1) == 0:
        return 0
    
    movie_2 = R[R.movieID==movie_id_2]
    movie_1_2 = pd.merge(movie_1,movie_2,how="inner",on="userID")
    
    return len(movie_1_2)/len(movie_1)


# In[296]:


@lru_cache
def calculate_advanced_association(movie_id_1,movie_id_2):    
    not_x_and_y = 0
    not_x = 0
    for user, data in R.groupby("userID"):
        movies = set(data.movieID)
        if (movie_id_1 not in movies) and (movie_id_2 in movies):
            not_x_and_y += 1
        if (movie_id_1 not in movies):
            not_x += 1
    if not_x_and_y == 0:
        return 0
    return calculate_simple_association(movie_id_1,movie_id_2)*(not_x/not_x_and_y)


# In[297]:


calculate_simple_association(661,1091)


# In[299]:


calculate_advanced_association(661,1091)


# In[305]:


def highest_simple(movie_id,n):
    movies = set(R.movieID)
    movies.remove(movie_id)
    retval = []
    for i in movies:
        retval.append((i,calculate_simple_association(movie_id,i),))
    retval = sorted(retval,key=lambda x:(-x[1],-x[0]))[:n]
    return pd.merge(pd.DataFrame(retval, columns={"movieID","simple"}),I, how="inner",on="movieID")


# In[306]:


highest_simple(661,10)


# In[307]:


def highest_advanced(movie_id,n):
    movies = set(R.movieID)
    movies.remove(movie_id)
    retval = []
    for i in movies:
        retval.append((i,calculate_advanced_association(movie_id,i),))
    retval = sorted(retval,key=lambda x:(-x[1],-x[0]))[:n]
    return pd.merge(pd.DataFrame(retval, columns={"movieID","simple"}),I, how="inner",on="movieID")


# In[292]:


highest_advanced(661,10)


# In[289]:


def get_most_rated(n):
    return pd.merge(R.movieID.value_counts().head(n).reset_index().rename(columns={"index":"movieID","movieID":"freq"}),I,how="inner",on="movieID")

get_most_rated(10)


# In[288]:


def get_most_rated_with_least_rating(n,least=4):
    return pd.merge(R[R.rating >= least].movieID.value_counts().head(n).reset_index().rename(columns={"index":"movieID","movieID":"freq"}),I,how="inner",on="movieID")

get_most_rated_with_least_rating(10,4)


# # Analysis

# Consider the movie with ID 1. What is the value of the simple product association for the
# movie with ID 1064?

# In[228]:


calculate_simple_association(1,1064)


# Consider the movie with ID 1. What is the value of the advanced product association for the
# movie with ID 1064?

# In[245]:


calculate_advanced_association(1,1064)


# Explain the difference between the values of the simple and advanced product association
# (question 1 and 2). How do we have to interpret these numbers?

# Answer:
# 
# - Simple association is an asymmetrical relationship that expresses the percentage of people who bought X, who also bought Y. X is something that occurred, what is now the probability that Y occurs?
# 
# - advanced product association calculates if X makes Y more likely than other products. 

# Consider the movie with ID 1. What is the value of the simple product association for the
# movie with ID 2858?

# In[230]:


calculate_simple_association(1,2858)


# What are the movie titles and genres of movies with ID 1, 1064, and 2858?

# In[322]:


I[(I.movieID == 1) | (I.movieID == 1064)| (I.movieID == 2858)| (I.movieID == 3941) ]


# Compare the results of question 1 and 4. Which movie has the highest simple association
# value (1064 or 2858) with movie ID 1 and explain why?
# 
# - answer: The movie with id 2858, American Beauty (1999), has a higher association. This is because the term "X and Y" is higher. This means that if people 

# Consider the movie with ID 1. What is the value of the advanced product association for the
# movie with ID 2858?

# In[273]:


calculate_advanced_association(1,2858)


# Compare the results of question 2 and 7. Which movie has the highest advanced association
# value (1064 or 2858) with movie ID 1 and explain why?

# 

# Calculate the top 10 most frequently rated movies. Provide the movie ID, number of users
# who rated the movie, and title for each.

# In[320]:


get_most_rated(10)


# Now you have to find the movies that can be associated to the movie with id 3941. Calculate
# the Top 5 movies with the highest simple association value. Provide the movie ID,
# association value, and title for each.

# In[308]:


highest_simple(3941,5)


# For the same movie as in question 10 (id 3941), calculate the Top 5 movies with the highest
# advanced association value. Provide the movie ID, association value, and title for each.

# In[309]:


highest_advanced(3941,5)


# Compare the resulting lists of question 9 and 10. What do you witness and how can you
# explain this?

# Compare the resulting lists of question 10 and 11. Which one is the best (=most accurate)
# according to you, and why?

# Recalculate the Top 10 most frequently rated movies. But use only ratings of at least 4 stars.
# Provide the movie ID, number of users who rated the movie, and title for each.

# In[321]:


get_most_rated_with_least_rating(10,4)


# Compare the resulting lists of question 9 and 14. What do you witness and how can you
# explain this?

# In[ ]:




