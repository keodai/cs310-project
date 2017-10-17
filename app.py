import os # Filesystem navigation.
import subprocess # For call to ffmpeg script.
import librosa # For feature extraction.
import librosa.display
import matplotlib.pyplot as plt # For graphs.

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
    y = [0] * len(file_list)
    sr = [0] * len(file_list)
    mfcc = [0] * len(file_list)

    for i, file in enumerate(file_list):
        y[i], sr[i] = librosa.load(file)
        mfcc[i] = librosa.feature.mfcc(y=y[i], sr=sr[i])
        if PLOT_RESULTS:
            plt.figure(num=i, figsize=(10, 4))
            librosa.display.specshow(mfcc[i], x_axis="time")
            plt.colorbar()
            plt.title("MFCC " + str(i))
            plt.tight_layout()
            plt.savefig(plot_path + "temp" + str(i) + ".png")


def main():
    mp3_to_wav()
    feature_extraction()


if __name__ == '__main__':
    main()