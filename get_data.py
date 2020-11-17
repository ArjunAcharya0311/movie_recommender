import pandas as pd


df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')


df1.columns = ['id','title','cast','crew']
df2 = df2.merge(df1, on='id')

df2.to_csv("movie_list.csv", index = False)
