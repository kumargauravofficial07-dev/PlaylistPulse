# PlaylistPulse
Spotify Track Intelligence System

Put your CSV at:
backend/data/raw/spotify_tracks.csv

Run `run_eda.bat` first.
Run `run_backend.bat` to start FastAPI.
Run `run_frontend.bat` to start React.

Backend docs: http://127.0.0.1:8000/docs
Frontend: http://localhost:5173

## Explore songs and get a prediction

Open the frontend while both the backend and frontend are running. In **Use your data**:

1. Search by song or artist, or choose a genre.
2. Select a song to fill the audio features from the processed dataset, or enter the feature values from a spreadsheet row yourself.
3. Select **Show track results** to see estimated popularity, the popular/not-popular prediction, and its audio group.

The song list is ordered by the dataset's popularity score. This starter dataset does not contain personal Spotify play counts, so these results cannot identify which songs you personally listen to most.
