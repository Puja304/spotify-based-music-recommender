import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#load dataset
songs = pd.read_csv("audio-added-Puja.csv", header=0, parse_dates=['timestamp'])
print(songs.columns)

#drop columns that are not needed + any na entries 
songs = songs.drop(['artist_kaggle', 'join_key', 'Unnamed: 0', 'track_kaggle', 'track_id', 'year'], axis=1)
songs = songs.dropna()


#looking at counts for each type of feature:
print(songs.dtypes)

#examine the variety in genres since it is not a numerical value but a categorical one
num_genres = songs['genre'].nunique()
print(f"Number of unique genres: {num_genres}")

# verify they are valid names
genre_list = songs['genre'].unique()
print("Genres:")
for genre in genre_list:
    print(f"- {genre}")

# the names are valid. for future clustering + other classifiers, add two options:

#one hot encode:
encoded = pd.get_dummies(songs['genre'], prefix='genre')
onehot = songs.copy()
onehot = onehot.drop(columns='genre')
onehot = pd.concat([onehot, encoded], axis=1)
onehot.to_csv("songs-cleaned-Puja.csv", index = False)


# #label encode:
# labeled = songs.copy()
# le = LabelEncoder()
# labeled['genre_label'] = le.fit_transform(labeled['genre'])
# labeled = labeled.drop(columns='genre')
# labeled.to_csv("songs-cleaned-label-Alia.csv", index = False)


#do some basic plotting to find overacrching trends

cols_to_plot = [
    'popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_ms_kaggle', 'time_signature'
]

# Plot histograms
songs[cols_to_plot].hist(bins=30, figsize=(16, 12), grid=False, edgecolor='black')
plt.suptitle('Distribution of Audio Features', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.show()

