import pandas as pd

# load streaming history
history = pd.read_json("StreamingHistory-Alia.json")
history = history.rename(columns={
    "endTime": "timestamp",
    "artistName": "artist",
    "trackName": "track",
    "msPlayed": "duration_ms"
})

# normalize
history["join_key"] = (history["artist"] + " - " + history["track"]).str.lower().str.strip()

# load Kaggle dataset
kaggle = pd.read_csv("spotify_data.csv")
kaggle = kaggle.rename(columns={
    "artist_name": "artist",
    "track_name": "track"
})

#normalize
kaggle["join_key"] = (kaggle["artist"] + " - " + kaggle["track"]).str.lower().str.strip()

# Merge on track name + artist name combination (normalized)
merged = pd.merge(history, kaggle, on="join_key", how="inner", suffixes=('_history', '_kaggle'))

# Save result
merged.to_csv("audio-added-Alia.csv", index=False)
print(f"✅ Merged and saved {len(merged)} songs to enriched_history_joined.csv")
