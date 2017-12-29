import setup
import paths

import sys
import os
from sklearn.externals import joblib
import logging
from scipy.spatial import distance


def main(args):
    logging.basicConfig(filename='logs/output.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

    if (not os.path.isfile("data/classifier.pkl")
            or not os.path.isfile("data/song_data.pkl")
            or not os.path.isfile("data/scaler.pkl")):
        setup.classify()

    clf = joblib.load("data/classifier.pkl")
    song_data = joblib.load("data/song_data.pkl")
    scaler = joblib.load("data/scaler.pkl")

    path = args[0]
    if not os.path.exists(path):
        if len(args) > 0:
            print("Specified path does not exist")
        else:
            print("One argument is required - the path to song file or directory to use")
    else:
        predicted = None
        features = None
        if os.path.isfile(path):
            song = setup.single_song(path, paths.output_dir)
            [song.normalised_features] = scaler.transform([song.features])
            [song.predicted_genre] = clf.predict([song.normalised_features])
            features, predicted = song.normalised_features, song.predicted_genre
        elif os.path.isdir(path):
            songs = setup.convert_and_get_data(path, paths.output_dir)
            normalised = []
            for song in songs:
                [song.normalised_features] = scaler.transform([song.features])
                [song.predicted_genre] = clf.predict([song.normalised_features])
                normalised.append(song.normalised_features)
            features = [float(sum(l)) / len(l) for l in zip(*normalised)]
            [predicted] = clf.predict([features])
        else:
            print("Path does not point to a file or directory")

        matches = []
        for training_song in song_data:
            if training_song.predicted_genre == predicted:
                dist = distance.euclidean(training_song.features, features)
                matches.append((training_song, dist))
        sorted_recommendations = sorted(matches, key=lambda l: l[1])
        print(sorted_recommendations)


if __name__ == '__main__':
    main(sys.argv[1:])