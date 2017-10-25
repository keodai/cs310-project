import os  # Filesystem navigation.
import subprocess  # For call to ffmpeg script.
import librosa  # For feature extraction.
import librosa.display  # For plots.
import matplotlib.pyplot as plt  # For plots.
import numpy as np  # For list processing - summary statistics of features in vector.
from sklearn import preprocessing
from tinytag import TinyTag  # For extracting metadata.

PLOT_MFCC_RESULTS = False

src_path = "/Users/matthew/Documents/university/CS310-Third-Year-Project/cs310-Project/samples/src/"
dest_path = "/Users/matthew/Documents/university/CS310-Third-Year-Project/cs310-Project/samples/dest/"
plot_path = "/Users/matthew/Documents/university/CS310-Third-Year-Project/cs310-Project/samples/plots/"
src_ext = ".mp3"
dest_ext = ".wav"

filepaths = []
genres = []
titles_and_artists = []
normalised_features = []


def files_in_dir(src_path):
    return [file for file in os.listdir(src_path) if not file.startswith('.')]


def mp3_to_wav():
    for file in files_in_dir(src_path):
        name = file[:file.rfind(".")]
        src = src_path + name + src_ext
        dest = dest_path + name + dest_ext
        filepaths.append(dest)
        genres.append(genre_from_metadata(src))
        titles_and_artists.append(title_and_artist_from_metadata(src))
        subprocess.call(["ffmpeg", "-i", src, dest])


def genre_from_metadata(src):
    return TinyTag.get(src).genre.replace('\x00', '')


def title_and_artist_from_metadata(src):
    return [
        TinyTag.get(src).title.replace('\x00', ''),
        TinyTag.get(src).artist.replace('\x00', ''),
        TinyTag.get(src).album.replace('\x00', '')
    ]


def plot_mfcc(mfcc, i):
    if PLOT_MFCC_RESULTS:
        plt.figure(num=i, figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis="time")
        plt.colorbar()
        plt.title("MFCC " + str(i))
        plt.tight_layout()
        plt.savefig(plot_path + "mfcc" + str(i) + ".png")


def feature_extraction():
    ysr = []
    stfeatures = []

    for i, file in enumerate(filepaths):
        ysr.append(librosa.load(file))
        zcr = librosa.feature.zero_crossing_rate(ysr[i][0])
        sc = librosa.feature.spectral_centroid(y=ysr[i][0], sr=ysr[i][1])
        sr = librosa.feature.spectral_rolloff(y=ysr[i][0], sr=ysr[i][1])
        sb = librosa.feature.spectral_bandwidth(y=ysr[i][0], sr=ysr[i][1])
        mfcc = librosa.feature.mfcc(y=ysr[i][0], sr=ysr[i][1])
        # Vector of mean and variance of features. Depending on performance, mean can be sum(l)/float(len(l)).
        # Var param  ddof=1 for n-1 instead of n. sum([(xi - m)**2 for xi in results]) / (len(results) - 1)
        feature_vector = [
            np.mean(zcr), np.var(zcr),
            np.mean(sc), np.var(sc),
            np.mean(sr), np.var(sr),
            np.mean(sb), np.var(sb),
            np.mean(mfcc), np.var(mfcc)
        ]
        print(str(feature_vector) + " " + str(genres[i]))
        stfeatures.append(feature_vector)
        plot_mfcc(mfcc, i)
    return preprocessing.minmax_scale(stfeatures)


def main():
    mp3_to_wav()
    normalised_features = feature_extraction()
    print(str(normalised_features))


if __name__ == '__main__':
    main()
