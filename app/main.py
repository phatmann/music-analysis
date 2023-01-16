import logging
import os
import tempfile
from enum import Enum, unique
from pathlib import Path
from typing import Dict, List, Union

import ffmpeg
import pinecone
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pinecone.core.client.model.query_response import QueryResponse
from pydantic import BaseModel
from tqdm import tqdm

from music_analysis.feature_extraction import extract_features

logger = logging.getLogger()
data_dir = Path("./covers32k")
api_key = os.getenv("PINECONE_API_KEY") or "820713d5-37c7-4570-877a-b23efb701b1c"
pinecone.init(api_key=api_key, environment="us-west1-gcp")
cover_song_index = pinecone.Index(index_name="cover-song")


class PineConeVector(BaseModel):
    id: str
    score: float
    metadata: Dict = {}
    sparseValues: Dict = {}
    values: List = []


class PineConeQueryResult(BaseModel):
    matches: List[PineConeVector] = []
    namespace: str = ""


class SearchResult(BaseModel):
    results: List[Dict] = []
    error: str = ""


@unique
class SearchOption(str, Enum):
    Auto = "Auto"
    Lyrics = "Lyrics"
    Mood = "Mood"
    Instrument = "Instrument"
    Genre = "Genre"


app = FastAPI(
    title="Music Search Service",
    version="0.1.0",
    contact={
        "email": "dut.hww@gmail.com",
    },
)
app.add_middleware(CORSMiddleware, allow_origins=["*"])


def search_cover_song(upload_files: List[UploadFile]) -> List[Dict]:
    results = []
    for file in upload_files:
        fd, tmp = tempfile.mkstemp()
        try:
            with open(tmp, "wb") as f:
                file.file.seek(0)
                content = file.file.read()
                f.write(content)
            query_features = extract_features(tmp)
        finally:
            os.unlink(tmp)
        result = cover_song_index.query(
            vector=query_features["embedding"].tolist(), top_k=1, include_metadata=True
        )
        results.append(result.to_dict())
    return results


@app.post("/search", summary="Search similar songs", response_model=SearchResult)
def search(
    files: List[UploadFile] = File(description="Multiple files as UploadFile"),
    option: SearchOption = SearchOption.Auto,
):
    ret = {}
    if option == SearchOption.Lyrics:
        try:
            results = search_cover_song(files)
            ret["results"] = results
        except Exception as ex:
            logger.error(ex)
            ret["error"] = str(ex)
    else:
        ret["error"] = f"Search option {SearchOption} is not supported"
    return ret


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3001)
