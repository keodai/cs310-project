import os # Filesystem navigation.
import subprocess # For call to ffmpeg script.
import librosa # For feature extraction.
import librosa.display
import matplotlib.pyplot as plt # For graphs.
import numpy as np
from scipy.io import wavfile

PLOT_RESULTS = True

src_path = "/Users/matthew/Documents/university/CS310-Third-Year-Project/cs310-Project/samples/src/"
dest_path = "/Users/matthew/Documents/university/CS310-Third-Year-Project/cs310-Project/samples/dest/"
plot_path = "/Users/matthew/Documents/university/CS310-Third-Year-Project/cs310-Project/samples/plots/"
src_ext = ".mp3"
dest_ext = ".wav"

file_list = []


def mp3_to_wav():
    for file in os.listdir(src_path):
        name = file[:file.rfind(".")]
        src = src_path + name + src_ext
        dest = dest_path + name + dest_ext
        file_list.append(dest)
        subprocess.call(["ffmpeg", "-i", src, dest])


def feature_extraction():
    ysr, stfeatures = ([] for i in range(2))
    #y, sr, mfcc, zcr, scent, sband, sroll = [0] * len(file_list)

    for i, file in enumerate(file_list):
        ysr.append(librosa.load(file))
        stfeatures.append(librosa.feature.zero_crossing_rate(ysr[i][0]))
        stfeatures.append(librosa.feature.spectral_centroid(y=ysr[i][0], sr=ysr[i][1]))
        stfeatures.append(librosa.feature.spectral_rolloff(y=ysr[i][0], sr=ysr[i][1]))
        stfeatures.append(librosa.feature.spectral_bandwidth(y=ysr[i][0], sr=ysr[i][1]))
        stfeatures.append(librosa.feature.mfcc(y=ysr[i][0], sr=ysr[i][1]))
        if PLOT_RESULTS:
            plt.figure(num=i, figsize=(10, 4))
            librosa.display.specshow(stfeatures[4], x_axis="time")
            plt.colorbar()
            plt.title("MFCC " + str(i))
            plt.tight_layout()
            plt.savefig(plot_path + "temp" + str(i) + ".png")


def main():
    mp3_to_wav()
    feature_extraction()


if __name__ == '__main__':
    main()