import setup
import paths

import sys
import os
from sklearn.externals import joblib
import logging
from scipy.spatial import distance
# from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KDTree


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
        output = []
        if mode == "SVM":  # Sorted songs in genre region.
            logging.info("SVM")
            matches = []
            for song in song_data:
                if song.predicted_genre == predicted:
                    dist = distance.euclidean(song.normalised_features, norm_features)
                    matches.append((song, dist))
            sorted_recommendations = sorted(matches, key=lambda l: l[1])[:10]
            for rec in sorted_recommendations:
                recommendation = rec[0]
                logging.info(recommendation)
                output.append(make_song_record(recommendation.title, recommendation.artist, recommendation.album, recommendation.src))
        elif mode == "FASTKMEANS":  # Unsorted songs in single cluster.
            logging.info("FASTKMEANS")
            [nearest_cluster] = kmeans.predict([norm_features])
            recommendations = [entry[0] for entry in zip(song_data, kmeans.labels_) if entry[1] == nearest_cluster]
            for recommendation in recommendations:
                logging.info(recommendation)
                output.append(make_song_record(recommendation.title, recommendation.artist, recommendation.album, recommendation.src))
        elif mode == "FASTSORTEDKMEANS":  # Sorted songs in single cluster.
            logging.info("FASTSORTEDKMEANS")
            [nearest_cluster] = kmeans.predict([norm_features])
            songs_in_cluster = [entry[0] for entry in zip(song_data, kmeans.labels_) if entry[1] == nearest_cluster]
            dist = []
            for sic in songs_in_cluster:
                dist.append(distance.euclidean(sic.normalised_features, norm_features))
            recommendations = sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])
            for rec in recommendations:
                recommendation = rec[0]
                logging.info(recommendation)
                output.append(make_song_record(recommendation.title, recommendation.artist, recommendation.album, recommendation.src))
        elif mode == "KMEANS":  # All clusters, sorted by cluster, then song distance.
            logging.info("KMEANS")
            recommendations = []
            cluster_label = -2
            song_cluster_ids = zip(song_data, kmeans.labels_)
            cluster_distances = zip(kmeans.predict(kmeans.cluster_centers_),
                                    [distance.euclidean(cluster_center, norm_features) for cluster_center in
                                     kmeans.cluster_centers_])
            sorted_cluster_distances = sorted(cluster_distances, key=lambda l: l[1])
            for entry in sorted_cluster_distances:
                if cluster_label != entry[0]:
                    cluster_label = entry[0]
                    songs_in_cluster = [entry[0] for entry in song_cluster_ids if
                                        entry[1] == cluster_label]
                    dist = []
                    for sic in songs_in_cluster:
                        dist.append(distance.euclidean(sic.normalised_features, norm_features))
                    recommendations.extend(sorted(zip(songs_in_cluster, dist), key=lambda l: l[1]))
            for rec in recommendations:
                recommendation = rec[0]
                logging.info(recommendation)
                output.append(make_song_record(recommendation.title, recommendation.artist, recommendation.album, recommendation.src))
        elif mode == "DBSCAN":  # Sorted songs in single cluster.
            logging.info("DBSCAN")
            # How to deal with noise?
            core_sample_labels = [dbscan.labels_[index] for index in dbscan.core_sample_indices_]
            core_samples = list(zip(core_sample_labels, dbscan.components_))
            song_cluster_ids = zip(song_data, dbscan.labels_)
            tree = KDTree(dbscan.components_)
            [[index]] = tree.query([norm_features])[1]
            nearest_cluster = core_samples[index][0]  # For single nearest cluster, change query for more clusters.
            songs_in_cluster = [entry[0] for entry in song_cluster_ids if entry[1] == nearest_cluster]
            dist = []
            for sic in songs_in_cluster:
                dist.append(distance.euclidean(sic.normalised_features, norm_features))
            recommendations = sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])
            for rec in recommendations:
                recommendation = rec[0]
                logging.info(recommendation)
                output.append(make_song_record(recommendation.title, recommendation.artist, recommendation.album, recommendation.src))
        elif mode == "SVM+KMEANS":
            logging.info("SVM+KMEANS")
            [genre_clusters] = [i[1] for i in genre_kmeans if i[0] == predicted]
            songs_in_genre = [song for song in song_data if song.predicted_genre == predicted]
            recommendations = []
            cluster_label = -2
            song_cluster_ids = zip(songs_in_genre, genre_clusters.labels_)
            cluster_distances = zip(genre_clusters.predict(genre_clusters.cluster_centers_),
                                    [distance.euclidean(cluster_center, norm_features) for cluster_center in
                                     genre_clusters.cluster_centers_])
            sorted_cluster_distances = sorted(cluster_distances, key=lambda l: l[1])
            for entry in sorted_cluster_distances:
                if cluster_label != entry[0]:
                    cluster_label = entry[0]
                    songs_in_cluster = [entry[0] for entry in song_cluster_ids if
                                        entry[1] == cluster_label]
                    dist = []
                    for sic in songs_in_cluster:
                        dist.append(distance.euclidean(sic.normalised_features, norm_features))
                    recommendations.extend(sorted(zip(songs_in_cluster, dist), key=lambda l: l[1]))
            for rec in recommendations:
                recommendation = rec[0]
                logging.info(recommendation)
                output.append(make_song_record(recommendation.title, recommendation.artist, recommendation.album, recommendation.src))
        elif mode == "SVM+DBSCAN":
            logging.info("SVM+DBSCAN")
            [genre_clusters] = [i[1] for i in genre_dbscan if i[0] == predicted]
            songs_in_genre = [song for song in song_data if song.predicted_genre == predicted]
            core_sample_labels = [dbscan.labels_[index] for index in dbscan.core_sample_indices_]
            core_samples = list(zip(core_sample_labels, dbscan.components_))
            song_cluster_ids = zip(songs_in_genre, genre_clusters.labels_)
            tree = KDTree(genre_clusters.components_)
            [[index]] = tree.query([norm_features])[1]
            nearest_cluster = core_samples[index][0]  # For single nearest cluster, change query for more clusters.
            songs_in_cluster = [entry[0] for entry in song_cluster_ids if entry[1] == nearest_cluster]
            dist = []
            for sic in songs_in_cluster:
                dist.append(distance.euclidean(sic.normalised_features, norm_features))
            recommendations = sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])
            for rec in recommendations:
                recommendation = rec[0]
                logging.info(recommendation)
                output.append(make_song_record(recommendation.title, recommendation.artist, recommendation.album, recommendation.src))
        elif mode == "DBSCAN+SVM":
            logging.info("DBSCAN+SVM")
            [predicted_cluster] = svm_on_dbscan.predict([norm_features])
            songs_in_cluster = [song for song in song_data if song.dbscan_cluster_id == predicted_cluster]
            matches = []
            for song in songs_in_cluster:
                dist = distance.euclidean(song.normalised_features, norm_features)
                matches.append((song, dist))
            sorted_recommendations = sorted(matches, key=lambda l: l[1])[:10]
            for rec in sorted_recommendations:
                recommendation = rec[0]
                logging.info(recommendation)
                output.append(make_song_record(recommendation.title, recommendation.artist, recommendation.album, recommendation.src))
        else:
            print("Invalid mode. Options: [SVM, FASTKMEANS, FASTSORTEDKMEANS, KMEANS, DBSCAN, SVM+KMEANS, SVM+DBSCAN, DBSCAN+SVM]")

        return output
        # END RECOMMENDATION


if __name__ == "__main__":
    recommend(sys.argv[1:])
