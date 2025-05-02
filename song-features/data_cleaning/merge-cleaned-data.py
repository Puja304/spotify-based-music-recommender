# combine the cleaned datasets into final ones to use for machine learning
import pandas as pd

#load files
weather = pd.read_csv("weather-cleaned.csv", header=0, parse_dates=['datetime'])
songs = pd.read_csv("songs-cleaned-Puja.csv", header=0, parse_dates=['timestamp'])

#ensure date formatting is consistent
songs['date'] = pd.to_datetime(songs['timestamp']).dt.date
songs = songs.drop(columns='timestamp')
weather['date'] = pd.to_datetime(weather['datetime']).dt.date
weather = weather.drop(columns='datetime')

songs_merged = pd.merge(weather, songs, on='date', how='inner')

songs_merged.to_csv("Merged-Puja.csv", index=False)
