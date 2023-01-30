import os
import tempfile

import ffmpeg
import numpy as np
import whisper
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

# 'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium',
# 'large-v1', 'large-v2', 'large'
whisper_model = whisper.load_model("tiny.en")
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
            .audio.filter("atrim", duration=duration)
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


class LyricsFeatures(BaseModel):
    lyrics: str
    embedding: np.ndarray


app = FastAPI(
    title="Music Feature Extraction Service",
    version="0.1.0",
    contact={
        "email": "dut.hww@gmail.com",
    },
)
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.post(
    "/extract_lyrics_features",
    summary="Extract lyrics features",
    response_model=LyricsFeatures,
)
def search(
    file: UploadFile = File(description="Multiple files as UploadFile"),
):
    fd, tmp = tempfile.mkstemp()
    try:
        with open(tmp, "wb") as f:
            file.file.seek(0)
            content = file.file.read()
            f.write(content)
        features = extract_features(tmp)
    finally:
        os.unlink(tmp)
    return features


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3002)
