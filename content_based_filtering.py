import pandas as pd
import numpy as np

df = pd.read_csv("movie_list.csv")

# Content-Based collaborative filtering the content of the movie is 
# used to find similarities with other movies


# based on plot description
# we need to convert the word vector of each overview
# compute Term Frequency - Inverse Document Frequency (TD-IDF)
df['overview']

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

df['overview'] = df['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(df['overview'])

tfidf_matrix.shape

# We can now compute a similarity score - euclidean, cosine simi, Pearson
# cosine similarity

from sklearn.metrics.pairwise import linear_kernel

cosine_simi = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['title_x']).drop_duplicates()

def get_recommendations(title, cosine_sim = cosine_simi):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df['title_x'].iloc[movie_indices]


get_recommendations("The Dark Knight Rises")

# Using actors, crew and keywords associated with genre

from ast import literal_eval

features = ['cast','crew','keywords','genres']
for feature in features:
    df[feature] = df[feature].apply(literal_eval)
    
    
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

    
def get_list(x):
    if isinstance(x, list) :
        names = [i['name'] for i in x]
        
        if len(names) > 3:
            names = names[:3]
        return names
    return []

df['director'] = df['crew'].apply(get_director)
features = ['cast', 'keywords','genres']

for feature in features:
    df[feature] = df[feature].apply(get_list)
    
df[['title_x','cast','director','keywords','genres']].head(5)

# convert names and keywords into lower case and strip spaces between them

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ","")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ",""))
        else:
            return ''

features = ['cast','keywords','director','genres']

for feature in features:
    df[feature] = df[feature].apply(clean_data)
    
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

df['soup'] = df.apply(create_soup, axis = 1)


from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words= 'english')
count_matrix = count.fit_transform(df['soup'])


from sklearn.metrics.pairwise import cosine_similarity

cosine_similar = cosine_similarity(count_matrix, count_matrix)

df = df.reset_index(drop = True)
indices = pd.Series(df.index, index = df['title_x'])

get_recommendations('The Dark Knight Rises', cosine_similar)
get_recommendations('The Godfather', cosine_similar)


# recommendation captures more metadata and has given better recommendation.