import sys
import json
import pandas as pd
import requests
import base64


#setup spotify
def get_spotify_token(client_id, client_secret):
    """Fetch an access token from Spotify API."""
    auth = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
    url = "https://accounts.spotify.com/api/token"
    headers = {"Authorization": f"Basic {auth}"}
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, headers=headers, data=data)
    return response.json().get("access_token")


#use it to find genres
def findGenreSpotify(artist_name):
    client_id = "92725123c83549a88b46c6200f1e6113"
    client_secret = "a53b585c8faf4233a8b084a3a8b7b00c"

    access_token = get_spotify_token(client_id, client_secret)
    headers = {"Authorization": f"Bearer {access_token}"}

    search_url = f"https://api.spotify.com/v1/search?q=artist:{artist_name}&type=artist&limit=1"
    response = requests.get(search_url, headers=headers)
    search_data = response.json()

    if "artists" in search_data and search_data["artists"]["items"]:
        artist_id = search_data["artists"]["items"][0]["id"]

        artist_url = f"https://api.spotify.com/v1/artists/{artist_id}"
        artist_response = requests.get(artist_url, headers=headers)
        artist_data = artist_response.json()

        genres = artist_data.get("genres", [])
        return genres[0] if genres else None
    return None

#falback 1
def findGenreLastFM(artist_name):
    API_KEY = "f2f61bfb7e031fb37796be456f762ad0"
    BASE_URL = "http://ws.audioscrobbler.com/2.0/"
    params = {
        "method": "artist.getTopTags",
        "api_key": API_KEY,
        "artist": artist_name,
        "format": "json"
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "toptags" in data and "tag" in data["toptags"]:
        tags = [tag["name"] for tag in data["toptags"]["tag"]]
        return tags[0] if tags else None
    return None


#fallback 2
def findGenreMusicBrainz(artist_name):
    """Get genre from MusicBrainz API."""
    BASE_URL = "https://musicbrainz.org/ws/2/"
    
    # Set up the headers with User-Agent
    headers = {
        "User-Agent": "MusicPrediction/1.0 (puja.shah321@gmail.com)"
    }

    params = {
        "query": f'artist:"{artist_name}"',
        "fmt": "json"
    }

    response = requests.get(BASE_URL + "artist/", params=params, headers=headers)
    data = response.json()

    if "artists" in data and data["artists"]:
        artist = data["artists"][0]
        if "genres" in artist:
            return artist["genres"][0]["name"]
    return None


#overall function
def findGenre(artist_name):
    """Try to get genre from Spotify, then Last.fm, then MusicBrainz."""
    genre = findGenreSpotify(artist_name)
    if genre:
        return genre

    genre = findGenreLastFM(artist_name)
    if genre:
        return genre

    return findGenreMusicBrainz(artist_name)


# Load JSON file with Spotify data
spotify_data = sys.argv[1]

with open(spotify_data, "r", encoding="utf-8") as file:
    data = json.load(file)

df = pd.DataFrame(data)

for i in range(14):
    df_slice = df.iloc[500*i:500*(i+1)].copy()
    df_slice["genre"] = df_slice["artistName"].apply(findGenre)
    
    # If it's the first batch, write the header; otherwise, skip it.
    if i == 0:
        df_slice.to_csv("genres.csv", mode='w', header=True, index=False)  # Write header only for the first batch
    else:
        df_slice.to_csv("genres.csv", mode='a', header=False, index=False)  # Append without header for subsequent batches
