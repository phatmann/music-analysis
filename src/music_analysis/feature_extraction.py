import logging
import os

import ffmpeg
import librosa
import numpy as np
import pandas as pd
from scipy import stats

from music_analysis import utils

whisper_model = None
sentence_emb_model = None
logger = logging.getLogger(__name__)


def load_audio(file: str, sr: int = 16000, duration: float = 30):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .audio.filter("atrim", duration=duration)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def extract_features(filename):
    global whisper_model, sentence_emb_model
    if whisper_model is None:
        import whisper

        whisper_model = whisper.load_model("large-v2")
    if sentence_emb_model is None:
        from sentence_transformers import SentenceTransformer

        sentence_emb_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    audio = load_audio(str(filename))
    result = whisper_model.transcribe(audio=audio)
    lyrics = result["text"]
    print(f"Song: {filename}, lyrics: {lyrics}")
    return {"lyrics": lyrics, "embedding": sentence_emb_model.encode(lyrics)}


def momentum_columns():
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


def compute_momentum_features(
    filepath: str, offset: float = 0.0, duration: float = None
):
    """Compute momentum spectral features"""
    features = pd.Series(index=momentum_columns(), dtype=np.float32, name=filepath)

    def feature_stats(name, values):
        features[name, "mean"] = np.mean(values, axis=1)
        features[name, "std"] = np.std(values, axis=1)
        features[name, "skew"] = stats.skew(values, axis=1)
        features[name, "kurtosis"] = stats.kurtosis(values, axis=1)
        features[name, "median"] = np.median(values, axis=1)
        features[name, "min"] = np.min(values, axis=1)
        features[name, "max"] = np.max(values, axis=1)

    try:
        x, sr = librosa.load(
            filepath, sr=None, mono=True, offset=offset, duration=duration
        )  # kaiser_fast

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


def spectral_columns():
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


def compute_features(filepath: str):
    features = pd.Series(index=spectral_columns(), dtype=np.float32)

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
