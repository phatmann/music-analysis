#!/usr/bin/python3
import logging
import multiprocessing
import os
import pickle
import random
import warnings
from pathlib import Path

import dotenv
import librosa
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from tqdm import tqdm

from music_analysis import utils

logger = logging.getLogger()

warnings.simplefilter("ignore", UserWarning)

dotenv.load_dotenv()


def columns():
    feature_sizes = dict(
        chroma_stft=12,
        chroma_cqt=12,
        chroma_cens=12,
        tonnetz=6,
        mfcc=20,
        rmse=1,
        zcr=1,
        spectral_centroid=1,
        spectral_bandwidth=1,
        spectral_contrast=7,
        spectral_rolloff=1,
    )
    moments = ("mean", "std", "skew", "kurtosis", "median", "min", "max")

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, "{:02d}".format(i + 1)) for i in range(size))
            columns.extend(it)

    names = ("feature", "statistics", "number")
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


def compute_features(filepath: str):

    features = pd.Series(index=columns(), dtype=np.float32, name=filepath)

    def feature_stats(name, values):
        features[name, "mean"] = np.mean(values, axis=1)
        features[name, "std"] = np.std(values, axis=1)
        features[name, "skew"] = stats.skew(values, axis=1)
        features[name, "kurtosis"] = stats.kurtosis(values, axis=1)
        features[name, "median"] = np.median(values, axis=1)
        features[name, "min"] = np.min(values, axis=1)
        features[name, "max"] = np.max(values, axis=1)

    try:
        x, sr = librosa.load(filepath, sr=None, mono=True)  # kaiser_fast

        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        feature_stats("zcr", f)

        cqt = np.abs(
            librosa.cqt(
                x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None
            )
        )
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(len(x) / 512) <= cqt.shape[1] <= np.ceil(len(x) / 512) + 1

        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats("chroma_cqt", f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats("chroma_cens", f)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats("tonnetz", f)

        del cqt
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
        del x

        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feature_stats("chroma_stft", f)

        f = librosa.feature.rms(S=stft)
        feature_stats("rmse", f)

        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats("spectral_centroid", f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        feature_stats("spectral_bandwidth", f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats("spectral_contrast", f)
        f = librosa.feature.spectral_rolloff(S=stft)
        feature_stats("spectral_rolloff", f)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats("mfcc", f)

    except Exception as e:
        logger.error("%s: %s", filepath, e)

    return features


def cluster(tracks):
    METADATA_DIR = Path(os.environ.get("METADATA_DIR")).expanduser()
    tracks = utils.load(METADATA_DIR / "tracks.csv")
    small_tracks = tracks[(tracks["set", "subset"] == "small")]

    features = utils.load(METADATA_DIR / "features.csv")

    X = features.loc[small_tracks.index, :].values

    kmeans = KMeans(n_clusters=100, random_state=0).fit(X)
    kmeans.labels_
    kmeans.cluster_centers_

    pickle.dump(kmeans, open("save.pkl", "wb"))


def predict(song: str):
    model = pickle.load(open("save.pkl", "rb"))
    features = compute_features(song)
    return model.predict(features.values.reshape(1, -1))


def predict_list(songs: list):
    METADATA_DIR = Path(os.environ.get("METADATA_DIR")).expanduser()
    tracks = utils.load(METADATA_DIR / "tracks.csv")
    small_tracks = tracks[(tracks["set", "subset"] == "small")]
    model = pickle.load(open("save.pkl", "rb"))
    small_tracks["cluster"] = model.labels_
    features = pd.DataFrame(index=songs, columns=columns(), dtype=np.float32)
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
    songs = Path("/media/hr/My Passport/music-dataset/fma_small/000").glob("*.mp3")
    results = predict_list([str(song) for song in random.sample(list(songs), 10)])
    print(results)
