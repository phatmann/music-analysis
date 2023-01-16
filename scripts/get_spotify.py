import os

import dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

dotenv.load_dotenv()

sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
track = sp.track("5ppcig5J8zvHej6P2qnBTt")

results = sp.search(q="weezer", limit=20)
for idx, track in enumerate(results["tracks"]["items"]):
    print(idx, track["name"])
