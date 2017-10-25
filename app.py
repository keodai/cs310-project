import os # Filesystem navigation.
import subprocess # For call to ffmpeg script.
import librosa # For feature extraction.
import librosa.display
import matplotlib.pyplot as plt # For graphs.
import numpy as np
from scipy.io import wavfile
from tinytag import TinyTag

PLOT_MFCC_RESULTS = False

src_path = "/Users/matthew/Documents/university/CS310-Third-Year-Project/cs310-Project/samples/src/"
dest_path = "/Users/matthew/Documents/university/CS310-Third-Year-Project/cs310-Project/samples/dest/"
plot_path = "/Users/matthew/Documents/university/CS310-Third-Year-Project/cs310-Project/samples/plots/"
src_ext = ".mp3"
dest_ext = ".wav"

filepaths = []
genres = []
titles_and_artists = []


def mp3_to_wav():
    for file in os.listdir(src_path):
        name = file[:file.rfind(".")]
        src = src_path + name + src_ext
        dest = dest_path + name + dest_ext
        filepaths.append(dest)
        subprocess.call(["ffmpeg", "-i", src, dest])
        genre_from_metadata(src)
        title_and_artist_from_metadata(src)


def genre_from_metadata(src):
    TinyTag.get(src).genre.replace('\x00', '')


def title_and_artist_from_metadata(src):
    titles_and_artists.append([
        TinyTag.get(src).title.replace('\x00', ''),
        TinyTag.get(src).artist.replace('\x00', ''),
        TinyTag.get(src).album.replace('\x00', '')
    ])

def feature_extraction():
    ysr = []
    stfeatures = []
    # y, sr, mfcc, zcr, scent, sband, sroll = [0] * len(file_list)
    # lists = [[] for i in xrange(num_lists)]

    for i, file in enumerate(filepaths):
        ysr.append(librosa.load(file))
        feature_vector = []
        feature_vector.append(librosa.feature.zero_crossing_rate(ysr[i][0]))
        feature_vector.append(librosa.feature.spectral_centroid(y=ysr[i][0], sr=ysr[i][1]))
        feature_vector.append(librosa.feature.spectral_rolloff(y=ysr[i][0], sr=ysr[i][1]))
        feature_vector.append(librosa.feature.spectral_bandwidth(y=ysr[i][0], sr=ysr[i][1]))
        feature_vector.append(librosa.feature.mfcc(y=ysr[i][0], sr=ysr[i][1]))
        # scale/normalise
        stfeatures.append(feature_vector)
        if PLOT_MFCC_RESULTS:
            plt.figure(num=i, figsize=(10, 4))
            librosa.display.specshow(stfeatures[i][4], x_axis="time")
            plt.colorbar()
            plt.title("MFCC " + str(i))
            plt.tight_layout()
            plt.savefig(plot_path + "mfcc" + str(i) + ".png")


def main():
    mp3_to_wav()
    feature_extraction()


if __name__ == '__main__':
    main()