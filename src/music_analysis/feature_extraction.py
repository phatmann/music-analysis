import numpy as np
import ffmpeg
import whisper
from sentence_transformers import SentenceTransformer, util

whisper_model = whisper.load_model("large-v2")
sentence_emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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
            .audio.filter('atrim', duration=duration)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def extract_features(filename):
    audio = load_audio(str(filename))
    result = whisper_model.transcribe(audio=audio)
    lyrics = result["text"]
    print(f"Song: {filename}, lyrics: {lyrics}")
    return {"lyrics": lyrics, "embedding": sentence_emb_model.encode(lyrics)}


