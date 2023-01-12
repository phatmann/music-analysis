from pathlib import Path
import essentia.standard as estd
from essentia.pytools.spectral import hpcpgram

data_dir = Path("./coversongs/covers32k")

def main():
    query_filename = str(data_dir / "Yesterday/en_vogue+Funky_Divas+09-Yesterday.mp3")
    true_ref_filename = str(data_dir / "Yesterday/beatles+1+11-Yesterday.mp3")
    false_ref_filename = str(
        data_dir / "Come_Together/aerosmith+Live_Bootleg+06-Come_Together.mp3"
    )

    # query cover song
    query_audio = estd.MonoLoader(filename=query_filename, sampleRate=32000)()
    # true cover
    true_cover_audio = estd.MonoLoader(filename=true_ref_filename, sampleRate=32000)()
    # wrong match
    false_cover_audio = estd.MonoLoader(filename=false_ref_filename, sampleRate=32000)()


    # compute frame-wise hpcp with default params
    query_hpcp = hpcpgram(query_audio, sampleRate=32000)

    true_cover_hpcp = hpcpgram(true_cover_audio, sampleRate=32000)

    false_cover_hpcp = hpcpgram(false_cover_audio, sampleRate=32000)

    cross_similarity = estd.ChromaCrossSimilarity(
        frameStackSize=9, frameStackStride=1, binarizePercentile=0.095, oti=True
    )


    true_pair_sim_matrix = cross_similarity(query_hpcp, true_cover_hpcp)


    false_pair_sim_matrix = cross_similarity(query_hpcp, false_cover_hpcp)

    cover_song_similarity = estd.CoverSongSimilarity(
        disOnset=0.5, disExtension=0.5, alignmentType="serra09", distanceType="asymmetric"
    )

    true_pair_score_matrix, true_pair_distance = cover_song_similarity(true_pair_sim_matrix)


    false_pair_score_matrix, false_pair_distance = cover_song_similarity(
        false_pair_sim_matrix
    )
    print(f'true pair distance {true_pair_distance}')
    print(f'false pair distance {false_pair_distance}')

if __name__ == '__main__':
    main()
