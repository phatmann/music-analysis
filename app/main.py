import os
import tempfile
from tqdm import tqdm
from pathlib import Path
import ffmpeg
import pinecone
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from pydantic import BaseModel

from pinecone.core.client.model.query_response import QueryResponse
from music_analysis.feature_extraction import extract_features

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
    namespace: str = ''

app = FastAPI(
    title="Music Search Service",
    version="0.1.0",
    contact={
        "email": "dut.hww@gmail.com",
    },
)
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.post("/search", summary="Search cover song", response_model=PineConeQueryResult)
def search_cover_song(file: UploadFile):
    fd, tmp = tempfile.mkstemp()
    try:
        with open(tmp, 'wb') as f:
            file.file.seek(0)
            content = file.file.read()
            f.write(content)
        query_features = extract_features(tmp)
    finally:
        os.unlink(tmp)
    results = cover_song_index.query(
        vector=query_features["embedding"].tolist(), top_k=5, include_metadata=True
    )
    return results.to_dict()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3001)
