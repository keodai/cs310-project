import os                                           # Filesystem navigation.
import subprocess                                   # For call to ffmpeg script.
import librosa                                      # For feature extraction.
import librosa.display                              # For plots.
import matplotlib.pyplot as plt                     # For plots.
import numpy as np                                  # For list processing - summary statistics of features in vector.
import sklearn                                      # For pre-processing and classifier.
from sklearn.preprocessing import MinMaxScaler      # For scaling of extracted features.
import logging                                      # For recording progress for debugging.
from sklearn.externals import joblib                # Saving stuff.
import os.path                                      # Check path

from tinytag import TinyTag                         # For extracting metadata.

PLOT_MFCC_RESULTS = False

training_src_path = "/Volumes/expansion/project/data/src/training/"
training_dest_path = "/Volumes/expansion/project/data/dest/training/"
test_src_path = "/Volumes/expansion/project/data/src/test/"
test_dest_path = "/Volumes/expansion/project/data/dest/test/"
plot_path = "/Volumes/expansion/project/data/plots/"
src_ext = ".mp3"
dest_ext = ".wav"

filepaths = []
test_filepaths = []
genres = []
test_listed_genre = []
titles_and_artists = []
test_titles_and_artists = []


def xstr(s):
    return '' if s is None else str(s)


def files_in_dir(src_path):
    return [file for file in os.listdir(src_path) if not file.startswith('.')]


def mp3_to_wav_training(src_path, src_ext, dest_path, dest_ext):
    for file in files_in_dir(src_path):
        name = file[:file.rfind(".")]
        src = src_path + name + src_ext
        dest = dest_path + name + dest_ext
        filepaths.append(dest)
        genres.append(genre_from_metadata(src))
        titles_and_artists.append(title_and_artist_from_metadata(src))
        subprocess.call(["ffmpeg", "-y", "-i", src, dest])


def mp3_to_wav_test(src_path, src_ext, dest_path, dest_ext):
    for file in files_in_dir(src_path):
        name = file[:file.rfind(".")]
        src = src_path + name + src_ext
        dest = dest_path + name + dest_ext
        test_filepaths.append(dest)
        test_listed_genre.append(genre_from_metadata(src))
        test_titles_and_artists.append(title_and_artist_from_metadata(src))
        subprocess.call(["ffmpeg", "-y", "-i", src, dest])


def genre_from_metadata(src):
    return xstr(TinyTag.get(src).genre).replace('\x00', '')


def title_and_artist_from_metadata(src):
    return [
        xstr(TinyTag.get(src).title).replace('\x00', ''),
        xstr(TinyTag.get(src).artist).replace('\x00', ''),
        xstr(TinyTag.get(src).album).replace('\x00', '')
    ]


def plot_mfcc(mfcc, i):
        plt.figure(num=i, figsize=(10, 4))
        librosa.display.specshow(mfcc, x_axis="time")
        plt.colorbar()
        plt.title("MFCC " + str(i))
        plt.tight_layout()
        plt.savefig(plot_path + "mfcc" + str(i) + ".png")


def feature_extraction(filepaths):
    ysr = []
    stfeatures = []

    for i, file in enumerate(filepaths):
        ysr.append(librosa.load(file))
        zcr = librosa.feature.zero_crossing_rate(ysr[i][0])
        sc = librosa.feature.spectral_centroid(y=ysr[i][0], sr=ysr[i][1])
        sr = librosa.feature.spectral_rolloff(y=ysr[i][0], sr=ysr[i][1])
        sb = librosa.feature.spectral_bandwidth(y=ysr[i][0], sr=ysr[i][1])
        mfcc = librosa.feature.mfcc(y=ysr[i][0], sr=ysr[i][1])
        feature_vector = [
            np.mean(zcr), np.var(zcr),
            np.mean(sc), np.var(sc),
            np.mean(sr), np.var(sr),
            np.mean(sb), np.var(sb),
            np.mean(mfcc), np.var(mfcc)
        ]
        logging.info(str(i) + ": " + str(feature_vector))
        stfeatures.append(feature_vector)
        if PLOT_MFCC_RESULTS:
            plot_mfcc(mfcc, i)
    return stfeatures


def main():
    logging.basicConfig(filename='output.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
    logging.info("TRAINING *******************************************************************************************")
    logging.info("CONVERTING FILES...")
    mp3_to_wav_training(training_src_path, src_ext, training_dest_path, dest_ext)
    logging.info("GENRES =============================================================================================")
    logging.info(str(genres))
    logging.info("FEATURE EXTRACTION =================================================================================")

    if os.path.isfile("features.pkl"):
        stfeatures = joblib.load("features.pkl")
    else:
        stfeatures = feature_extraction(filepaths)
        joblib.dump(stfeatures, 'features.pkl')

    scaler = MinMaxScaler()
    scaler.fit(stfeatures)
    normalised_features = scaler.transform(stfeatures)
    logging.info("NORMALISED FEATURE VECTORS =========================================================================")
    logging.info(str(normalised_features))

    if os.path.isfile("classifier.pkl"):
        clf = joblib.load("classifier.pkl")
    else:
        # Create an SVM classifier.
        clf = sklearn.svm.SVC()
        # Train the classifier.
        logging.info("Training classifier...")
        clf.fit(normalised_features, genres)
        joblib.dump(clf, 'classifier.pkl')

    logging.info("Training complete.")
    logging.info("TESTING *******************************************************************************************")

    logging.info("CONVERTING FILES...")
    mp3_to_wav_test(test_src_path, src_ext, test_dest_path, dest_ext)

    logging.info("GENRES =============================================================================================")
    logging.info(str(test_listed_genre))

    logging.info("FEATURE EXTRACTION =================================================================================")

    if os.path.isfile("test_features.pkl"):
        test_features = joblib.load("test_features.pkl")
    else:
        test_features = feature_extraction(test_filepaths)
        joblib.dump(test_features, 'test_features.pkl')

    normalised_test_features = scaler.transform(test_features)
    logging.info("NORMALISED FEATURE VECTORS =========================================================================")
    logging.info(str(normalised_test_features))


    # Predict genre using extracted feature data
    predicted_genres = clf.predict(test_features)

    # Compare predicted genre to listed genre - Retrieve listed genre from metadata - dictionary?
    logging.info("ANALYSIS *******************************************************************************************")
    non_matches = [(i,j) for i, j in zip(predicted_genres, test_listed_genre) if i != j]
    logging.info("Genre conflicts: " + str(non_matches))
    accuracy = len(non_matches)/len(predicted_genres)
    logging.info("Accuracy: " + str(accuracy) + "%")
    # Return songs in same genre (hashmap normalised? feature vector -> metadata + filepath)

    # Calculate closest songs (distance between feature vectors? - support vectors?)

    # Return songs (Json format for UI?)


if __name__ == '__main__':
    main()