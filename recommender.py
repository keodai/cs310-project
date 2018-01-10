import setup
import paths

import sys
import os
from sklearn.externals import joblib
import logging
from scipy.spatial import distance
from sklearn.neighbors import KDTree

MAX_RECS = 100


def fast_kmeans(kmeans, norm_features, song_data):
    [nearest_cluster] = kmeans.predict([norm_features])
    songs_labels = [entry for entry in zip(song_data, kmeans.labels_) if entry[1] == nearest_cluster]
    return nearest_cluster, songs_labels


def calculate_distances(songs_in_cluster, norm_features):
    dist = []
    for sic in songs_in_cluster:
        dist.append(distance.euclidean(sic.normalised_features, norm_features))
    return dist


def perform_dbscan(dbscan, song_data, norm_features):
    # How to deal with noise?
    core_sample_labels = [dbscan.labels_[index] for index in dbscan.core_sample_indices_]
    core_samples = list(zip(core_sample_labels, dbscan.components_))
    song_cluster_ids = zip(song_data, dbscan.labels_)
    tree = KDTree(dbscan.components_)
    [[index]] = tree.query([norm_features])[1]
    nearest_cluster = core_samples[index][0]  # For single nearest cluster, change query for more clusters.
    songs_in_cluster = [entry[0] for entry in song_cluster_ids if entry[1] == nearest_cluster]
    dist = calculate_distances(songs_in_cluster, norm_features)
    return sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])[:MAX_RECS]


def perform_kmeans(kmeans, song_data, norm_features):
    recs = []
    cluster_label = -2
    song_cluster_ids = zip(song_data, kmeans.labels_)
    cluster_distances = zip(kmeans.predict(kmeans.cluster_centers_),
                            [distance.euclidean(cluster_center, norm_features) for cluster_center in
                             kmeans.cluster_centers_])
    sorted_cluster_distances = sorted(cluster_distances, key=lambda l: l[1])
    for entry in sorted_cluster_distances:
        if cluster_label != entry[0]:
            cluster_label = entry[0]
            songs_in_cluster = [entry[0] for entry in song_cluster_ids if entry[1] == cluster_label]
            dist = calculate_distances(songs_in_cluster, norm_features)
            recs.extend(sorted(zip(songs_in_cluster, dist), key=lambda l: l[1]))
    return recs[:MAX_RECS]


def svm_then_classifier(genre_classifier, predicted, song_data, norm_features, perform_clustering):
    [genre_clusters] = [i[1] for i in genre_classifier if i[0] == predicted]
    songs_in_genre = [song for song in song_data if song.predicted_genre == predicted]
    return perform_clustering(genre_clusters, songs_in_genre, norm_features)


def make_song_record(title, artist, album, path):
    return {"title": title, "artist": artist, "album": album, "src": path}


def main(args):
    recommend(args)


def recommend(args):
    logging.basicConfig(filename="logs/output.log", level=logging.DEBUG, format="%(asctime)s %(message)s")
    required_files = ["data/song_data.pkl", "data/test_song_data.pkl", "data/scaler.pkl", "data/classifier.pkl",
                      "data/kmeans.pkl", "data/genre_kmeans.pkl", "data/dbscan.pkl", "data/svm_on_dbscan.pkl",
                      "data/genre_dbscan.pkl"]
    required_files_present = [os.path.isfile(file) for file in required_files]
    if not all(required_files_present):
        setup.create()

    song_data = joblib.load(required_files[0])
    test_song_data = joblib.load(required_files[1])
    scaler = joblib.load(required_files[2])
    clf = joblib.load(required_files[3])
    kmeans = joblib.load(required_files[4])
    genre_kmeans = joblib.load(required_files[5])
    dbscan = joblib.load(required_files[6])
    svm_on_dbscan = joblib.load(required_files[7])
    genre_dbscan = joblib.load(required_files[8])

    if len(args) != 2:
        print("Usage: python recommender path_to_music recommendation_mode")
    else:
        path = args[0]
        mode = args[1]
        predicted = None
        norm_features = None
        if os.path.isfile(path):
            song = setup.single_song(path, paths.output_dir)
            logging.info("Listed Genre: " + song.listed_genre)
            [song.normalised_features] = scaler.transform([song.features])
            [song.predicted_genre] = clf.predict([song.normalised_features])
            logging.info("Predicted Genre (SVM): " + song.predicted_genre)
            norm_features, predicted = song.normalised_features, song.predicted_genre
        elif os.path.isdir(path):
            songs = setup.convert_and_get_data(path, paths.output_dir)
            normalised = []
            for song in songs:
                [song.normalised_features] = scaler.transform([song.features])
                [song.predicted_genre] = clf.predict([song.normalised_features])
                normalised.append(song.normalised_features)
            norm_features = [float(sum(l)) / len(l) for l in zip(*normalised)]
            [predicted] = clf.predict([norm_features])
        else:
            print("Path does not point to a file or directory")

        # BEGIN RECOMMENDATION
        logging.info("Recommendations:")
        recommendations = []
        if mode == "SVM":  # Sorted songs in genre region.
            logging.info("SVM")
            songs_in_genre = [song for song in song_data if song.predicted_genre == predicted]
            dist = calculate_distances(songs_in_genre, norm_features)
            recommendations = sorted(zip(songs_in_genre, dist), key=lambda l: l[1])[:MAX_RECS]
        elif mode == "FASTKMEANS":  # Unsorted songs in single cluster.
            logging.info("FASTKMEANS")
            recommendations = fast_kmeans(kmeans, norm_features, song_data)[1][:MAX_RECS]
        elif mode == "FASTSORTEDKMEANS":  # Sorted songs in single cluster.
            logging.info("FASTSORTEDKMEANS")
            songs_in_cluster = [entry[0] for entry in fast_kmeans(kmeans, norm_features, song_data)[1]]
            dist = calculate_distances(songs_in_cluster, norm_features)
            recommendations = sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])[:MAX_RECS]
        elif mode == "KMEANS":  # All clusters, sorted by cluster, then song distance.
            logging.info("KMEANS")
            recommendations = perform_kmeans(kmeans, song_data, norm_features)
        elif mode == "DBSCAN":  # Sorted songs in single cluster.
            logging.info("DBSCAN")
            recommendations = perform_dbscan(dbscan, song_data, norm_features)
        elif mode == "SVM+KMEANS":
            logging.info("SVM+KMEANS")
            recommendations = svm_then_classifier(genre_kmeans, predicted, song_data, norm_features, perform_kmeans)
        elif mode == "SVM+DBSCAN":
            logging.info("SVM+DBSCAN")
            recommendations = svm_then_classifier(genre_dbscan, predicted, song_data, norm_features, perform_dbscan)
        elif mode == "DBSCAN+SVM":
            logging.info("DBSCAN+SVM")
            [predicted_cluster] = svm_on_dbscan.predict([norm_features])
            songs_in_cluster = [song for song in song_data if song.dbscan_cluster_id == predicted_cluster]
            dist = calculate_distances(songs_in_cluster, norm_features)
            recommendations = sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])[:MAX_RECS]
        else:
            print("Invalid mode. Options: [SVM, FASTKMEANS, FASTSORTEDKMEANS, KMEANS, DBSCAN, SVM+KMEANS, SVM+DBSCAN, DBSCAN+SVM]")

        output = []
        for rec in recommendations:
            recommendation = rec[0]
            logging.info(recommendation)
            output.append(
                make_song_record(recommendation.title, recommendation.artist, recommendation.album, recommendation.src))
        return output, predicted
        # END RECOMMENDATION


if __name__ == "__main__":
    recommend(sys.argv[1:])
