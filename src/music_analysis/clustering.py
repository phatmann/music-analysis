#!/usr/bin/python3
import logging
import os
import pickle
import random
import warnings
from pathlib import Path

import dotenv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from music_analysis import utils
from music_analysis.feature_extraction import compute_features, spectral_columns

logger = logging.getLogger()

warnings.simplefilter("ignore", UserWarning)

dotenv.load_dotenv()


def train():
    metadata_dir = utils.get_metadata_dir()
    # tracks = utils.load(METADATA_DIR / "tracks.csv")
    # small_tracks = tracks[(tracks["set", "subset"] == "small")]
    small_tracks = utils.load(metadata_dir / "small-tracks.csv")
    features = utils.load(metadata_dir / "small-track-features.csv")
    features = features.fillna(0)
    # invalid_tids = [99134, 108925, 133297]  # these tracks should be ignored

    X = features.loc[small_tracks.index, :].values

    kmeans = KMeans(n_clusters=100, random_state=0).fit(X)
    kmeans.labels_
    kmeans.cluster_centers_

    pickle.dump(kmeans, open("save.pkl", "wb"))
    logger.info("Saved clustering model")


def predict(song: str):
    model = pickle.load(open("save.pkl", "rb"))
    features = compute_features(song)
    return model.predict(features.values.reshape(1, -1))


def predict_list(songs: list):
    metadata_dir = utils.get_metadata_dir()
    small_tracks = utils.load(metadata_dir / "small-tracks.csv")

    model = pickle.load(open("save.pkl", "rb"))
    small_tracks["cluster"] = model.labels_
    features = pd.DataFrame(index=songs, columns=spectral_columns(), dtype=np.float32)
    for song in songs:
        features.loc[song] = compute_features(song)
    clusters = model.predict(features)

    # sample tracks
    total_samples = None
    for cluster in clusters:
        samples = small_tracks[small_tracks["cluster"] == cluster].sample(2)
        if total_samples is None:
            total_samples = samples.copy()
        else:
            total_samples = pd.concat([total_samples, samples])
    return total_samples


if __name__ == "__main__":
    # train()
    songs = Path("/media/hr/My Passport/music-dataset/fma_small/000").glob("*.mp3")
    random_10_songs = [str(song) for song in random.sample(list(songs), 10)]
    results = predict_list(random_10_songs)
    print(results)
