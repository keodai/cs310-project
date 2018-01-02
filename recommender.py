import setup
import paths

import sys
import os
from sklearn.externals import joblib
import logging
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances


def main(args):
    logging.basicConfig(filename='logs/output.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

    if (not os.path.isfile("data/classifier.pkl")
            or not os.path.isfile("data/song_data.pkl")
            or not os.path.isfile("data/scaler.pkl")
            or not os.path.isfile("data/clusters.pkl")):
        setup.classify()

    clf = joblib.load("data/classifier.pkl")
    song_data = joblib.load("data/song_data.pkl")
    scaler = joblib.load("data/scaler.pkl")
    clusters = joblib.load("data/clusters.pkl")

    if len(args) < 1:
        print("One argument is required - the path to song file or directory to use.")
    else:
        path = args[0]
        predicted = None
        features = None
        if os.path.isfile(path):
            song = setup.single_song(path, paths.output_dir)
            logging.info("Listed Genre: " + song.listed_genre)
            [song.normalised_features] = scaler.transform([song.features])
            [song.predicted_genre] = clf.predict([song.normalised_features])
            logging.info("Predicted Genre: " + song.predicted_genre)
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
        for song in song_data:
            if song.predicted_genre == predicted:
                dist = distance.euclidean(song.features, features)
                matches.append((song.src, dist))
        sorted_recommendations = sorted(matches, key=lambda l: l[1])[:10]
        logging.info("Recommendations:")
        for recommendation in sorted_recommendations:
            logging.info(recommendation)

        # With clustering
        dist_matrix = pairwise_distances([song.normalised_features for song in song_data])
        [genre_clusters] = [i[1] for i in clusters if i[0] == predicted]
        distances = []
        for cluster in genre_clusters:
            dist = dist_matrix[features, cluster].min()  # Single linkage
            distances.append(dist)
            print(distances)
            print("The cluster for {} is {}").format(features, cluster)
        # print(genre_clusters.fit_predict([features]))  # Not sure about this - need to find core points or use knn instead?
        # Select recommendations from this cluster
        # cluster_dict = {i: X[clusters.labels == i] for i in xrange(n_clusters_)}


if __name__ == '__main__':
    main(sys.argv[1:])
