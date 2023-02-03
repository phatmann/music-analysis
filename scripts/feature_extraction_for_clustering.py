#!/usr/bin/python3
import logging
import os
import warnings
from pathlib import Path

import dotenv
import numpy as np
import pandas as pd
from rich.progress import track

from music_analysis import utils
from music_analysis.feature_extraction import compute_features, spectral_columns

logger = logging.getLogger(__name__)

warnings.simplefilter("ignore", UserWarning)

dotenv.load_dotenv()


def create_library_features():
    """Calcualte features for song library and store them to csv"""
    metadata_dir = utils.get_metadata_dir()
    tracks = utils.load(metadata_dir / "tracks.csv")
    tracks = tracks[(tracks["set", "subset"] == "small")]
    features = pd.DataFrame(
        index=tracks.index, columns=spectral_columns(), dtype=np.float32
    )

    for i, tid in enumerate(track(tracks.index, description="Processing..."), start=1):
        filepath = utils.get_audio_path(os.environ.get("AUDIO_DIR"), tid)
        features.loc[tid] = compute_features(filepath)

        if i % 10 == 0:
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
