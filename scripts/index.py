import os
import numpy as np
import pinecone
from tqdm import tqdm
from pathlib import Path
from feature_extraction import extract_features

data_dir = Path("./covers32k")
song_library = {}  # song -> embedding
api_key = os.getenv("PINECONE_API_KEY") or "820713d5-37c7-4570-877a-b23efb701b1c"
pinecone.init(api_key=api_key, environment="us-west1-gcp")
cover_song_index = pinecone.Index(index_name="cover-song")


def load_song_embedding():
    library_file = data_dir / "songs.npz"
    if library_file.exists():
        songs = np.load(library_file, allow_pickle=True)
        song_library.update(songs)
    else:
        song_files = list(data_dir.glob("**/*.mp3"))
        pbar = tqdm(total=len(song_files))
        for song in song_files:
            print(f"Loading song {song}")
            features = extract_features(song)
            song_library[str(song)] = features
            pbar.update()
        np.savez(library_file, **song_library)
        print(f"Saved song library to {library_file}")


def main():
    load_song_embedding()
    cover_song_index.upsert(
        vectors=[
            (
                str(id),
                value.item()["embedding"].tolist(),
                {"filename": key.split("/")[-1], "lyrics": value.item()["lyrics"]},
            )
            for id, (key, value) in enumerate(song_library.items())
        ]
    )
    print(cover_song_index.describe_index_stats())


if __name__ == "__main__":
    main()
