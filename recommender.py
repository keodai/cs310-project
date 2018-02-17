import multi_logging
import setup
import paths

import sys
import os
from sklearn.externals import joblib
from scipy.spatial import distance
from sklearn.neighbors import KDTree
from timeit import default_timer as timer

reload(sys)
sys.setdefaultencoding('utf-8')

MAX_RECS = 100

timing = multi_logging.setup_logger('timing', 'logs/recommendation_times.log')
logging = multi_logging.setup_logger('output', 'logs/output.log')

vector_type = None


def fast_kmeans(kmeans, norm_features, song_data):
    [nearest_cluster] = kmeans.predict([norm_features])
    songs_labels = [entry for entry in zip(song_data, kmeans.labels_) if entry[1] == nearest_cluster]
    return nearest_cluster, songs_labels


def calculate_distances(songs_in_cluster, norm_features):
    dist = []
    for sic in songs_in_cluster:
        if vector_type == "TIMBRE":
            dist.append(distance.euclidean(sic.normalised_timbre, norm_features))
        elif vector_type == "MID":
            dist.append(distance.euclidean(sic.normalised_features, norm_features))
        elif vector_type == "TIMBRE_SQ":
            dist.append(distance.euclidean(sic.normalised_timbre_sq, norm_features))
        elif vector_type == "MID_SQ":
            dist.append(distance.euclidean(sic.normalised_features_sq, norm_features))
        else:
            print("Invalid vector type selected")
            exit(1)

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

    if vector_type == "TIMBRE":
        songs_in_genre = [song for song in song_data if song.predicted_genre_timbre == predicted]
    elif vector_type == "MID":
        songs_in_genre = [song for song in song_data if song.predicted_genre_features == predicted]
    elif vector_type == "TIMBRE_SQ":
        songs_in_genre = [song for song in song_data if song.predicted_genre_timbre_sq == predicted]
    elif vector_type == "MID_SQ":
        songs_in_genre = [song for song in song_data if song.predicted_genre_features_sq == predicted]
    else:
        print("Invalid vector type selected")
        exit(1)
    return perform_clustering(genre_clusters, songs_in_genre, norm_features)


def make_song_record(title, artist, album, path):
    return {"title": title, "artist": artist, "album": album, "src": path}


def main(args):
    recommend(args)


def recommend(args):
    global vector_type
    if len(args) != 3:
        print("Usage: python recommender path_to_music recommendation_mode vector_type")
    else:
        path = args[0]
        mode = args[1]
        vector_type = args[2]
        predicted = None
        norm_features = None
        predictions = []

        required_files = ["data/song_data.pkl", "data/test_song_data.pkl", "data/scaler_timbre.pkl", "data/classifier_timbre.pkl",
                          "data/kmeans_timbre.pkl", "data/genre_kmeans_timbre.pkl", "data/dbscan_timbre.pkl", "data/svm_on_dbscan_timbre.pkl",
                          "data/genre_dbscan_timbre.pkl", "data/scaler_mid.pkl", "data/classifier_mid.pkl",
                          "data/kmeans_mid.pkl", "data/genre_kmeans_mid.pkl", "data/dbscan_mid.pkl", "data/svm_on_dbscan_mid.pkl",
                          "data/genre_dbscan_mid.pkl", "data/scaler_timbre_sq.pkl", "data/classifier_timbre_sq.pkl",
                          "data/kmeans_timbre_sq.pkl", "data/genre_kmeans_timbre_sq.pkl", "data/dbscan_timbre_sq.pkl", "data/svm_on_dbscan_timbre_sq.pkl",
                          "data/genre_dbscan_timbre_sq.pkl", "data/scaler_mid_sq.pkl", "data/classifier_mid_sq.pkl",
                          "data/kmeans_mid_sq.pkl", "data/genre_kmeans_mid_sq.pkl", "data/dbscan_mid_sq.pkl", "data/svm_on_dbscan_mid_sq.pkl",
                          "data/genre_dbscan_mid_sq.pkl"]
        required_files_present = [os.path.isfile(file) for file in required_files]
        if not all(required_files_present):
            setup.create()
            # todo: error here

        song_data = joblib.load(required_files[0])
        test_song_data = joblib.load(required_files[1])

        if vector_type == "TIMBRE":
            scaler_timbre = joblib.load(required_files[2])
            # if mode == "SVM" or mode == "SVM+KMEANS" or mode == "SVM+DBSCAN":
            clf_timbre = joblib.load(required_files[3])
            if mode == "FASTKMEANS" or mode == "FASTSORTEDKMEANS" or mode == "KMEANS":
                kmeans_timbre = joblib.load(required_files[4])
            if mode == "SVM+KMEANS":
                genre_kmeans_timbre = joblib.load(required_files[5])
            if mode == "DBSCAN" or mode == "DBSCAN+SVM":
                dbscan_timbre = joblib.load(required_files[6])
            if mode == "DBSCAN+SVM":
                svm_on_dbscan_timbre = joblib.load(required_files[7])
            if mode == "SVM+DBSCAN":
                genre_dbscan_timbre = joblib.load(required_files[8])
        elif vector_type == "MID":
            scaler_mid = joblib.load(required_files[9])
            # if mode == "SVM" or mode == "SVM+KMEANS" or mode == "SVM+DBSCAN":
            clf_mid = joblib.load(required_files[10])
            if mode == "FASTKMEANS" or mode == "FASTSORTEDKMEANS" or mode == "KMEANS":
                kmeans_mid = joblib.load(required_files[11])
            if mode == "SVM+KMEANS":
                genre_kmeans_mid = joblib.load(required_files[12])
            if mode == "DBSCAN" or mode == "DBSCAN+SVM":
                dbscan_mid = joblib.load(required_files[13])
            if mode == "DBSCAN+SVM":
                svm_on_dbscan_mid = joblib.load(required_files[14])
            if mode == "SVM+DBSCAN":
                genre_dbscan_mid = joblib.load(required_files[15])
        elif vector_type == "TIMBRE_SQ":
            scaler_timbre_sq = joblib.load(required_files[16])
            # if mode == "SVM" or mode == "SVM+KMEANS" or mode == "SVM+DBSCAN":
            clf_timbre_sq = joblib.load(required_files[17])
            if mode == "FASTKMEANS" or mode == "FASTSORTEDKMEANS" or mode == "KMEANS":
                kmeans_timbre_sq = joblib.load(required_files[18])
            if mode == "SVM+KMEANS":
                genre_kmeans_timbre_sq = joblib.load(required_files[19])
            if mode == "DBSCAN" or mode == "DBSCAN+SVM":
                dbscan_timbre_sq = joblib.load(required_files[20])
            if mode == "DBSCAN+SVM":
                svm_on_dbscan_timbre_sq = joblib.load(required_files[21])
            if mode == "SVM+DBSCAN":
                genre_dbscan_timbre_sq = joblib.load(required_files[22])
        elif vector_type == "MID_SQ":
            scaler_mid_sq = joblib.load(required_files[23])
            # if mode == "SVM" or mode == "SVM+KMEANS" or mode == "SVM+DBSCAN":
            clf_mid_sq = joblib.load(required_files[24])
            if mode == "FASTKMEANS" or mode == "FASTSORTEDKMEANS" or mode == "KMEANS":
                kmeans_mid_sq = joblib.load(required_files[25])
            if mode == "SVM+KMEANS":
                genre_kmeans_mid_sq = joblib.load(required_files[26])
            if mode == "DBSCAN" or mode == "DBSCAN+SVM":
                dbscan_mid_sq = joblib.load(required_files[27])
            if mode == "DBSCAN+SVM":
                svm_on_dbscan_mid_sq = joblib.load(required_files[28])
            if mode == "SVM+DBSCAN":
                genre_dbscan_mid_sq = joblib.load(required_files[29])
        else:
            print("Invalid vector type selected")
            exit(1)

        warning = None

        if os.path.isfile(path):
            song = setup.single_song(path, paths.output_dir)
            logging.info("Listed Genre: " + song.listed_genre)

            if vector_type == "TIMBRE":
                [song.normalised_timbre] = scaler_timbre.transform([song.timbre])
                [song.predicted_genre_timbre] = clf_timbre.predict([song.normalised_timbre])
                logging.info("Predicted Genre (SVM): " + song.predicted_genre_timbre)
                norm_features, predicted = song.normalised_timbre, song.predicted_genre_timbre
            elif vector_type == "MID":
                [song.normalised_features] = scaler_mid.transform([song.features])
                [song.predicted_genre_features] = clf_mid.predict([song.normalised_features])
                logging.info("Predicted Genre (SVM): " + song.predicted_genre_features)
                norm_features, predicted = song.normalised_features, song.predicted_genre_features
            elif vector_type == "TIMBRE_SQ":
                [song.normalised_timbre_sq] = scaler_timbre_sq.transform([song.timbre_sq])
                [song.predicted_genre_timbre_sq] = clf_timbre_sq.predict([song.normalised_timbre_sq])
                logging.info("Predicted Genre (SVM): " + song.predicted_genre_timbre_sq)
                norm_features, predicted = song.normalised_timbre_sq, song.predicted_genre_timbre_sq
            elif vector_type == "MID_SQ":
                [song.normalised_features_sq] = scaler_mid_sq.transform([song.features_sq])
                [song.predicted_genre_features_sq] = clf_mid_sq.predict([song.normalised_features_sq])
                logging.info("Predicted Genre (SVM): " + song.predicted_genre_features_sq)
                norm_features, predicted = song.normalised_features_sq, song.predicted_genre_features_sq
            else:
                print("Invalid vector type selected")
                exit(1)

            predictions.append(predicted)
        elif os.path.isdir(path):
            songs = setup.convert_and_get_data(path, paths.output_dir)
            normalised = []
            for song in songs:
                if vector_type == "TIMBRE":
                    [song.normalised_timbre] = scaler_timbre.transform([song.timbre])
                    [song.predicted_genre_timbre] = clf_timbre.predict([song.normalised_timbre])
                    predictions.append(song.predicted_genre_timbre)
                    normalised.append(song.normalised_timbre)
                elif vector_type == "MID":
                    [song.normalised_features] = scaler_mid.transform([song.features])
                    [song.predicted_genre_features] = clf_mid.predict([song.normalised_features])
                    predictions.append(song.predicted_genre_features)
                    normalised.append(song.normalised_features)
                elif vector_type == "TIMBRE_SQ":
                    [song.normalised_timbre_sq] = scaler_timbre_sq.transform([song.timbre_sq])
                    [song.predicted_genre_timbre_sq] = clf_timbre_sq.predict([song.normalised_timbre_sq])
                    predictions.append(song.predicted_genre_timbre_sq)
                    normalised.append(song.normalised_timbre_sq)
                elif vector_type == "MID_SQ":
                    [song.normalised_features_sq] = scaler_mid_sq.transform([song.features_sq])
                    [song.predicted_genre_features_sq] = clf_mid_sq.predict([song.normalised_features_sq])
                    predictions.append(song.predicted_genre_features_sq)
                    normalised.append(song.normalised_features_sq)
                else:
                    print("Invalid vector type selected")
                    exit(1)
            norm_features = [float(sum(l)) / len(l) for l in zip(*normalised)]
            if vector_type == "TIMBRE":
                [predicted] = clf_timbre.predict([norm_features])
            elif vector_type == "MID":
                [predicted] = clf_mid.predict([norm_features])
            elif vector_type == "TIMBRE_SQ":
                [predicted] = clf_timbre_sq.predict([norm_features])
            elif vector_type == "MID_SQ":
                [predicted] = clf_mid_sq.predict([norm_features])
            else:
                print("Invalid vector type selected")
                exit(1)

            if predictions.count(predictions[0]) != len(predictions):
                warning = "Input songs are from different genres"
        else:
            print("Path does not point to a file or directory")

        # BEGIN RECOMMENDATION
        logging.info("Recommendations:")
        recommendations = []
        start = timer()
        if vector_type == "TIMBRE":
            if mode == "SVM":  # Sorted songs in genre region.
                logging.info("SVM")
                songs_in_genre = [song for song in song_data if song.predicted_genre_timbre == predicted]
                dist = calculate_distances(songs_in_genre, norm_features)
                recommendations = sorted(zip(songs_in_genre, dist), key=lambda l: l[1])[:MAX_RECS]
            elif mode == "FASTKMEANS":  # Unsorted songs in single cluster.
                logging.info("FASTKMEANS")
                recommendations = fast_kmeans(kmeans_timbre, norm_features, song_data)[1][:MAX_RECS]
            elif mode == "FASTSORTEDKMEANS":  # Sorted songs in single cluster.
                logging.info("FASTSORTEDKMEANS")
                songs_in_cluster = [entry[0] for entry in fast_kmeans(kmeans_timbre, norm_features, song_data)[1]]
                dist = calculate_distances(songs_in_cluster, norm_features)
                recommendations = sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])[:MAX_RECS]
            elif mode == "KMEANS":  # All clusters, sorted by cluster, then song distance.
                logging.info("KMEANS")
                recommendations = perform_kmeans(kmeans_timbre, song_data, norm_features)
            elif mode == "DBSCAN":  # Sorted songs in single cluster.
                logging.info("DBSCAN")
                recommendations = perform_dbscan(dbscan_timbre, song_data, norm_features)
            elif mode == "SVM+KMEANS":
                logging.info("SVM+KMEANS")
                recommendations = svm_then_classifier(genre_kmeans_timbre, predicted, song_data, norm_features, perform_kmeans)
            elif mode == "SVM+DBSCAN":
                logging.info("SVM+DBSCAN")
                recommendations = svm_then_classifier(genre_dbscan_timbre, predicted, song_data, norm_features, perform_dbscan)
            elif mode == "DBSCAN+SVM":
                logging.info("DBSCAN+SVM")
                [predicted_cluster] = svm_on_dbscan_timbre.predict([norm_features])
                songs_in_cluster = [song for song in song_data if song.dbscan_cluster_id_timbre == predicted_cluster]
                dist = calculate_distances(songs_in_cluster, norm_features)
                recommendations = sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])[:MAX_RECS]
            else:
                print(
                    "Invalid mode. Options: [SVM, FASTKMEANS, FASTSORTEDKMEANS, KMEANS, DBSCAN, SVM+KMEANS, SVM+DBSCAN, DBSCAN+SVM]")
        elif vector_type == "MID":
            if mode == "SVM":  # Sorted songs in genre region.
                logging.info("SVM")
                songs_in_genre = [song for song in song_data if song.predicted_genre_features == predicted]
                dist = calculate_distances(songs_in_genre, norm_features)
                recommendations = sorted(zip(songs_in_genre, dist), key=lambda l: l[1])[:MAX_RECS]
            elif mode == "FASTKMEANS":  # Unsorted songs in single cluster.
                logging.info("FASTKMEANS")
                recommendations = fast_kmeans(kmeans_mid, norm_features, song_data)[1][:MAX_RECS]
            elif mode == "FASTSORTEDKMEANS":  # Sorted songs in single cluster.
                logging.info("FASTSORTEDKMEANS")
                songs_in_cluster = [entry[0] for entry in fast_kmeans(kmeans_mid, norm_features, song_data)[1]]
                dist = calculate_distances(songs_in_cluster, norm_features)
                recommendations = sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])[:MAX_RECS]
            elif mode == "KMEANS":  # All clusters, sorted by cluster, then song distance.
                logging.info("KMEANS")
                recommendations = perform_kmeans(kmeans_mid, song_data, norm_features)
            elif mode == "DBSCAN":  # Sorted songs in single cluster.
                logging.info("DBSCAN")
                recommendations = perform_dbscan(dbscan_mid, song_data, norm_features)
            elif mode == "SVM+KMEANS":
                logging.info("SVM+KMEANS")
                recommendations = svm_then_classifier(genre_kmeans_mid, predicted, song_data, norm_features, perform_kmeans)
            elif mode == "SVM+DBSCAN":
                logging.info("SVM+DBSCAN")
                recommendations = svm_then_classifier(genre_dbscan_mid, predicted, song_data, norm_features, perform_dbscan)
            elif mode == "DBSCAN+SVM":
                logging.info("DBSCAN+SVM")
                [predicted_cluster] = svm_on_dbscan_mid.predict([norm_features])
                songs_in_cluster = [song for song in song_data if song.dbscan_cluster_id_mid == predicted_cluster]
                dist = calculate_distances(songs_in_cluster, norm_features)
                recommendations = sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])[:MAX_RECS]
            else:
                print("Invalid mode. Options: [SVM, FASTKMEANS, FASTSORTEDKMEANS, KMEANS, DBSCAN, SVM+KMEANS, SVM+DBSCAN, DBSCAN+SVM]")
        elif vector_type == "TIMBRE_SQ":
            if mode == "SVM":  # Sorted songs in genre region.
                logging.info("SVM")
                songs_in_genre = [song for song in song_data if song.predicted_genre_timbre_sq == predicted]
                dist = calculate_distances(songs_in_genre, norm_features)
                recommendations = sorted(zip(songs_in_genre, dist), key=lambda l: l[1])[:MAX_RECS]
            elif mode == "FASTKMEANS":  # Unsorted songs in single cluster.
                logging.info("FASTKMEANS")
                recommendations = fast_kmeans(kmeans_timbre_sq, norm_features, song_data)[1][:MAX_RECS]
            elif mode == "FASTSORTEDKMEANS":  # Sorted songs in single cluster.
                logging.info("FASTSORTEDKMEANS")
                songs_in_cluster = [entry[0] for entry in fast_kmeans(kmeans_timbre_sq, norm_features, song_data)[1]]
                dist = calculate_distances(songs_in_cluster, norm_features)
                recommendations = sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])[:MAX_RECS]
            elif mode == "KMEANS":  # All clusters, sorted by cluster, then song distance.
                logging.info("KMEANS")
                recommendations = perform_kmeans(kmeans_timbre_sq, song_data, norm_features)
            elif mode == "DBSCAN":  # Sorted songs in single cluster.
                logging.info("DBSCAN")
                recommendations = perform_dbscan(dbscan_timbre_sq, song_data, norm_features)
            elif mode == "SVM+KMEANS":
                logging.info("SVM+KMEANS")
                recommendations = svm_then_classifier(genre_kmeans_timbre_sq, predicted, song_data, norm_features, perform_kmeans)
            elif mode == "SVM+DBSCAN":
                logging.info("SVM+DBSCAN")
                recommendations = svm_then_classifier(genre_dbscan_timbre_sq, predicted, song_data, norm_features, perform_dbscan)
            elif mode == "DBSCAN+SVM":
                logging.info("DBSCAN+SVM")
                [predicted_cluster] = svm_on_dbscan_timbre_sq.predict([norm_features])
                songs_in_cluster = [song for song in song_data if song.dbscan_cluster_id_timbre_sq == predicted_cluster]
                dist = calculate_distances(songs_in_cluster, norm_features)
                recommendations = sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])[:MAX_RECS]
            else:
                print("Invalid mode. Options: [SVM, FASTKMEANS, FASTSORTEDKMEANS, KMEANS, DBSCAN, SVM+KMEANS, SVM+DBSCAN, DBSCAN+SVM]")
        elif vector_type == "MID_SQ":
            if mode == "SVM":  # Sorted songs in genre region.
                logging.info("SVM")
                songs_in_genre = [song for song in song_data if song.predicted_genre_features_sq == predicted]
                dist = calculate_distances(songs_in_genre, norm_features)
                recommendations = sorted(zip(songs_in_genre, dist), key=lambda l: l[1])[:MAX_RECS]
            elif mode == "FASTKMEANS":  # Unsorted songs in single cluster.
                logging.info("FASTKMEANS")
                recommendations = fast_kmeans(kmeans_mid_sq, norm_features, song_data)[1][:MAX_RECS]
            elif mode == "FASTSORTEDKMEANS":  # Sorted songs in single cluster.
                logging.info("FASTSORTEDKMEANS")
                songs_in_cluster = [entry[0] for entry in fast_kmeans(kmeans_mid_sq, norm_features, song_data)[1]]
                dist = calculate_distances(songs_in_cluster, norm_features)
                recommendations = sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])[:MAX_RECS]
            elif mode == "KMEANS":  # All clusters, sorted by cluster, then song distance.
                logging.info("KMEANS")
                recommendations = perform_kmeans(kmeans_mid_sq, song_data, norm_features)
            elif mode == "DBSCAN":  # Sorted songs in single cluster.
                logging.info("DBSCAN")
                recommendations = perform_dbscan(dbscan_mid_sq, song_data, norm_features)
            elif mode == "SVM+KMEANS":
                logging.info("SVM+KMEANS")
                recommendations = svm_then_classifier(genre_kmeans_mid_sq, predicted, song_data, norm_features, perform_kmeans)
            elif mode == "SVM+DBSCAN":
                logging.info("SVM+DBSCAN")
                recommendations = svm_then_classifier(genre_dbscan_mid_sq, predicted, song_data, norm_features, perform_dbscan)
            elif mode == "DBSCAN+SVM":
                logging.info("DBSCAN+SVM")
                [predicted_cluster] = svm_on_dbscan_mid_sq.predict([norm_features])
                songs_in_cluster = [song for song in song_data if song.dbscan_cluster_id_mid_sq == predicted_cluster]
                dist = calculate_distances(songs_in_cluster, norm_features)
                recommendations = sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])[:MAX_RECS]
            else:
                print("Invalid mode. Options: [SVM, FASTKMEANS, FASTSORTEDKMEANS, KMEANS, DBSCAN, SVM+KMEANS, SVM+DBSCAN, DBSCAN+SVM]")
        else:
            print("Invalid vector type selected")
            exit(1)

        end = timer()
        recommendation_time = end - start
        timing.info('Recommendation time ' + vector_type + ': ' + mode + ' - ' + str(recommendation_time))

        output = []
        for rec in recommendations:
            recommendation = rec[0]
            logging.info(recommendation)
            output.append(
                make_song_record(recommendation.title, recommendation.artist, recommendation.album, recommendation.src.replace(paths.project_audio_dir, "")))
        return output, predictions, warning
        # END RECOMMENDATION


if __name__ == "__main__":
    recommend(sys.argv[1:])
