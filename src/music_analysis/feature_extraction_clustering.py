#!/usr/bin/python3
import logging
import os
import warnings
from multiprocessing.pool import ThreadPool as Pool
from pathlib import Path

import dotenv
import librosa
import numpy as np
import pandas as pd
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

    columns = []
    for name, size in feature_sizes.items():
        it = ((name, "{:02d}".format(i + 1)) for i in range(size))
        columns.extend(it)

    names = ("feature", "number")
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


def compute_features(tid: str):
    filepath = utils.get_audio_path(os.environ.get("AUDIO_DIR"), tid)
    features = pd.Series(index=columns(), dtype=np.float32, name=tid)

    def feature_stats(name, values):
        # the get the features at the middle of the frame
        features[name] = values[:, int(values.shape[1] / 2)]

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


def create_library_features():
    """Calcualte features for song library and store them to csv"""
    METADATA_DIR = Path(os.environ.get("METADATA_DIR")).expanduser()
    tracks = utils.load(METADATA_DIR / "tracks.csv")
    tracks = tracks[(tracks["set", "subset"] == "small")]
    features = pd.DataFrame(index=tracks.index, columns=columns(), dtype=np.float32)

    # More than usable CPUs to be CPU bound, not I/O bound. Beware memory.
    nb_workers = int(1.5 * len(os.sched_getaffinity(0)))

    print("Working with {} processes.".format(nb_workers))

    pool = Pool(nb_workers)
    it = pool.imap_unordered(compute_features, tracks.index)

    for i, row in enumerate(tqdm(it, total=len(tracks.index)), start=1):
        features.loc[row.name] = row

        if i % 500 == 0:
            save(features, 10)

    save(features, 10)
    test(features, 10)


def save(features, ndigits):
    # Should be done already, just to be sure.
    features.sort_index(axis=0, inplace=True)
    features.sort_index(axis=1, inplace=True)

    features.to_csv("small-track-features.csv", float_format="%.{}e".format(ndigits))


def test(features, ndigits):
    indices = features[features.isnull().any(axis=1)].index
    if len(indices) > 0:
        print("Failed tracks: {}".format(", ".join(str(i) for i in indices)))

    tmp = utils.load("small-track-features.csv")
    np.testing.assert_allclose(tmp.values, features.values, rtol=10**-ndigits)


if __name__ == "__main__":
    create_library_features()
