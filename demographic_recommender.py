import pandas as pd
import numpy as np

df = pd.read_csv("movie_list.csv")

# mean rating across all movies
total_mean = df['vote_average'].mean()

# minimum votes required to be listed in the charts
# use 90th percentile as cutoff

minimum = df['vote_count'].quantile(0.9)

# filter movies that qualify for the chart

q_movies = df.copy().loc[df['vote_count'] >= minimum]

q_movies.shape

def weighted_rating(x, mini=minimum, tmean = total_mean):
    votes = x['vote_count']
    rating = x['vote_average']
    return (votes/(votes+mini) * rating) + (mini/(mini+votes) + tmean)


q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

q_movies = q_movies.sort_values('score', ascending = False)

q_movies[['title_x','vote_count','vote_average','score']]

# creates a general recommendation to all users, not sensitive to the interst 
# of particular users
