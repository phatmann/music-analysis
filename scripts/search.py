import numpy as np
from tqdm import tqdm
from pathlib import Path
import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram

data_dir = Path("./covers32k")
song_library = {}  # song name -> hpcp


def load_song_library():
    library_file = data_dir / "songs.npz"
    if library_file.exists():
        songs = np.load(library_file)
        song_library.update(songs)
    else:
        song_files = list(data_dir.glob("**/*.mp3"))
        pbar = tqdm(total=len(song_files))
        for song in song_files:
            print(f"Loading song {song}")
            audio = estd.MonoLoader(filename=str(song), sampleRate=32000)()
            hpcp = hpcpgram(audio, sampleRate=32000)
            song_library[str(song)] = hpcp
            pbar.update()
        np.savez(library_file, **song_library)
        print(f"Saved song library to {library_file}")


def search(query_filename):
    cross_similarity = estd.ChromaCrossSimilarity(
        frameStackSize=9, frameStackStride=1, binarizePercentile=0.095, oti=True
    )
    cover_song_similarity = estd.CoverSongSimilarity(
        disOnset=0.5,
        disExtension=0.5,
        alignmentType="serra09",
        distanceType="asymmetric",
    )

    query_audio = estd.MonoLoader(filename=query_filename, sampleRate=32000)()
    query_hpcp = hpcpgram(query_audio, sampleRate=32000)

    distances = []
    pbar = tqdm(total=len(song_library))
    for song, hpcp in song_library.items():
        score_matrix, pair_distance = cover_song_similarity(
            cross_similarity(query_hpcp, hpcp)
        )
        distances.append((song, pair_distance))
        print(f"Song {song}, distance {pair_distance}")
        pbar.update()
    distances = sorted(distances, key=lambda x: x[1])
    print("song\tdistance")
    for d in distances:
        print(f"{d[0]}\t{d[1]}")


def main():
    load_song_library()

    filename = str(data_dir / "Yesterday/en_vogue+Funky_Divas+09-Yesterday.mp3")
    search(filename)


if __name__ == "__main__":
    main()
