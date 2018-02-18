import multi_logging
import setup
import paths

import sys
import os
from scipy.spatial import distance
from sklearn.neighbors import KDTree
from timeit import default_timer as timer

# Set default encoding (Python 2.7)
reload(sys)
sys.setdefaultencoding('utf-8')

MAX_RECS = 100  # Maximum number of recommendations to return

# Loggers
timing = multi_logging.setup_logger('timing', 'logs/recommendation_times.log')
logging = multi_logging.setup_logger('output', 'logs/output.log')


# Retrieve nearest K-means cluster and contained songs
def fast_kmeans(kmeans, norm_features, song_data):
    [nearest_cluster] = kmeans.predict([norm_features])
    songs_labels = [entry for entry in zip(song_data, kmeans.labels_) if entry[1] == nearest_cluster]
    return nearest_cluster, songs_labels


# Calculate distances from input vector to other songs in cluster (Euclidean distance)
def calculate_distances(songs_in_cluster, norm_features, vector_type):
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


# Retrieve recommendations using DBSCAN
def perform_dbscan(dbscan, song_data, norm_features, vector_type):
    # How to deal with noise?
    core_sample_labels = [dbscan.labels_[index] for index in dbscan.core_sample_indices_]
    core_samples = list(zip(core_sample_labels, dbscan.components_))
    song_cluster_ids = zip(song_data, dbscan.labels_)
    tree = KDTree(dbscan.components_)
    [[index]] = tree.query([norm_features])[1]
    nearest_cluster = core_samples[index][0]  # For single nearest cluster, change query for more clusters.
    songs_in_cluster = [entry[0] for entry in song_cluster_ids if entry[1] == nearest_cluster]
    dist = calculate_distances(songs_in_cluster, norm_features, vector_type)
    return sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])[:MAX_RECS]


# Retrieve recommendations using K-means
def perform_kmeans(kmeans, song_data, norm_features, vector_type):
    recs = []
    cluster_label = -2
    song_cluster_ids = zip(song_data, kmeans.labels_)
    cluster_distances = zip(kmeans.predict(kmeans.cluster_centers_),
                            [distance.euclidean(cluster_center, norm_features)
                             for cluster_center in kmeans.cluster_centers_])
    sorted_cluster_distances = sorted(cluster_distances, key=lambda l: l[1])
    for entry in sorted_cluster_distances:
        if cluster_label != entry[0]:
            cluster_label = entry[0]
            songs_in_cluster = [entry[0] for entry in song_cluster_ids if entry[1] == cluster_label]
            dist = calculate_distances(songs_in_cluster, norm_features, vector_type)
            recs.extend(sorted(zip(songs_in_cluster, dist), key=lambda l: l[1]))
    return recs[:MAX_RECS]


# Uses SVM to predict genre, before using one of the clustering techniques within the genre
def svm_then_clustering(genre_classifier, predicted, song_data, norm_features, perform_clustering, vector_type):
    [genre_clusters] = [i[1] for i in genre_classifier if i[0] == predicted]
    songs_in_genre = [song for song in song_data if song.get_predicted_genre(vector_type) == predicted]
    return perform_clustering(genre_clusters, songs_in_genre, norm_features, vector_type)


# Create a dictionary record for a song
def make_song_record(title, artist, album, path):
    return {"title": title, "artist": artist, "album": album, "src": path}


def select_data(item, vector_type, data, condition=True, ):
    if condition:
        name = item + '_' + vector_type.lower()
        return data[name]
    else:
        return None


# Perform recommendation on selected song using specified mode and vector type
# Called from Flask endpoint to generate recommendations based on form input
def recommend(args):
    if len(args) != 4:
        raise ValueError("Usage: python recommender path_to_music recommendation_mode vector_type data")
    else:
        # Retrieve specified song location and options
        path = args[0]
        mode = args[1]
        vector_type = args[2]
        warning = None
        predictions = []

        # Select required files for the current recommendation task
        data = args[3]
        song_data = data['song_data']
        scaler = select_data('scaler', vector_type, data)
        svm_classifier = select_data('classifier', vector_type, data)
        kmeans = select_data('kmeans', vector_type, data, mode == "FASTKMEANS" or mode == "FASTSORTEDKMEANS" or mode == "KMEANS")
        genre_kmeans = select_data('genre_kmeans', vector_type, data, mode == "SVM+KMEANS")
        dbscan = select_data('dbscan', vector_type, data, mode == "DBSCAN" or mode == "DBSCAN+SVM")
        svm_on_dbscan = select_data('svm_on_dbscan', vector_type, data, mode == "DBSCAN+SVM")
        genre_dbscan = select_data('genre_dbscan', vector_type, data, mode == "SVM+DBSCAN")

        # Recommendations for single file (Not used by Flask)
        if os.path.isfile(path):
            song = setup.single_song(path, paths.output_dir)  # Input song conversion and feature extraction
            logging.info("Listed Genre: " + song.listed_genre)

            nf = scaler.transform([song.get_features(vector_type)])[0]
            song.set_normalised_features(vector_type, nf)
            pg = svm_classifier.predict([nf])[0]
            song.set_predicted_genre(vector_type, pg)
            logging.info("Predicted Genre (SVM): " + pg)
            norm_features, predicted = nf, pg
            predictions.append(predicted)

        # Directory
        elif os.path.isdir(path):
            songs = setup.convert_and_get_data(path, paths.output_dir)  # Input song conversion and feature extraction
            normalised = []
            for song in songs:
                nf = scaler.transform([song.get_features(vector_type)])[0]
                song.set_normalised_features(vector_type, nf)
                pg = svm_classifier.predict([nf])[0]
                song.set_predicted_genre(vector_type, pg)
                predictions.append(pg)
                normalised.append(nf)
            norm_features = [float(sum(l)) / len(l) for l in zip(*normalised)]  # Use the average of features as vector
            [predicted] = svm_classifier.predict([norm_features])
            if predictions.count(predictions[0]) != len(predictions):
                warning = "Input songs are from different genres"
        else:
            raise IOError('Path does not point to a file or directory')

        # Begin recommendation
        logging.info("Recommendations:")
        start = timer()
        if mode == "SVM":  # Sorted songs in genre region.
            logging.info("SVM")
            songs_in_genre = [song for song in song_data if song.get_predicted_genre(vector_type) == predicted]
            dist = calculate_distances(songs_in_genre, norm_features, vector_type)
            recommendations = sorted(zip(songs_in_genre, dist), key=lambda l: l[1])[:MAX_RECS]
        elif mode == "FASTKMEANS":  # Unsorted songs in single cluster.
            logging.info("FASTKMEANS")
            recommendations = fast_kmeans(kmeans, norm_features, song_data)[1][:MAX_RECS]
        elif mode == "FASTSORTEDKMEANS":  # Sorted songs in single cluster.
            logging.info("FASTSORTEDKMEANS")
            songs_in_cluster = [entry[0] for entry in fast_kmeans(kmeans, norm_features, song_data)[1]]
            dist = calculate_distances(songs_in_cluster, norm_features, vector_type)
            recommendations = sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])[:MAX_RECS]
        elif mode == "KMEANS":  # All clusters, sorted by cluster, then song distance.
            logging.info("KMEANS")
            recommendations = perform_kmeans(kmeans, song_data, norm_features, vector_type)
        elif mode == "DBSCAN":  # Sorted songs in single cluster.
            logging.info("DBSCAN")
            recommendations = perform_dbscan(dbscan, song_data, norm_features, vector_type)
        elif mode == "SVM+KMEANS":
            logging.info("SVM+KMEANS")
            recommendations = svm_then_clustering(genre_kmeans, predicted, song_data, norm_features, perform_kmeans, vector_type)
        elif mode == "SVM+DBSCAN":
            logging.info("SVM+DBSCAN")
            recommendations = svm_then_clustering(genre_dbscan, predicted, song_data, norm_features, perform_dbscan, vector_type)
        elif mode == "DBSCAN+SVM":
            logging.info("DBSCAN+SVM")
            [predicted_cluster] = svm_on_dbscan.predict([norm_features])
            songs_in_cluster = [song for song in song_data if song.get_dbscan_cluster_id(vector_type) == predicted_cluster]
            dist = calculate_distances(songs_in_cluster, norm_features, vector_type)
            recommendations = sorted(zip(songs_in_cluster, dist), key=lambda l: l[1])[:MAX_RECS]
        else:
            raise ValueError("Invalid mode. Options: [SVM, FASTKMEANS, FASTSORTEDKMEANS, KMEANS, DBSCAN, SVM+KMEANS, SVM+DBSCAN, DBSCAN+SVM]")

        end = timer()
        recommendation_time = end - start
        timing.info('Recommendation time ' + vector_type + ': ' + mode + ' - ' + str(recommendation_time))

        output = []
        for rec in recommendations:
            recommendation = rec[0]
            logging.info(recommendation)
            output.append(make_song_record(recommendation.title, recommendation.artist, recommendation.album, recommendation.src.replace(paths.project_audio_dir, "")))
        return output, predictions, warning


def main(args):
    recommend(args)


if __name__ == "__main__":
    recommend(sys.argv[1:])
