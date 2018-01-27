import utils
import converter
import paths
from song import Song

import os
import multi_logging
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
import sklearn
from timeit import default_timer as timer


# CONVERSION/NORMALISATION
# ------------------------
# SVM
# INDIVIDUAL K-MEANS
# GENRE K-MEANS
# DBSCAN
# DBSCAN ON SVM
# SVM ON DBSCAN

logging = multi_logging.setup_logger('output', 'logs/output.log')
timing = multi_logging.setup_logger('timing', 'logs/training_times.log')


def single_song(src, dst_path):
    logging.info("Processing: " + src)
    dst = converter.convert(src, dst_path, paths.dst_ext)
    return Song(src, dst)


def convert_and_get_data(src_path, dst_path):
    song_data = []
    for src in utils.visible(src_path):
        song_data.append(single_song(src, dst_path))
    return song_data


def calculate_or_load(filename, fun, *args):
    if os.path.isfile(filename):
        return joblib.load(filename)
    else:
        return fun(*args)


def songs_to_features(song_data):
    mid_features = []
    timbre_features = []
    mid_sq_features = []
    timbre_sq_features = []
    listed_genres = []
    for song in song_data:
        mid_features.append(song.features)
        timbre_features.append(song.timbre)
        mid_sq_features.append(song.features_sq)
        timbre_sq_features.append(song.timbre_sq)
        listed_genres.append(song.listed_genre)
    return mid_features, timbre_features, mid_sq_features, timbre_sq_features, listed_genres


def db_cluster(data):
    return sklearn.cluster.DBSCAN().fit(data)


def perform_scaling(features):
    scaler = MinMaxScaler()
    scaler.fit(features)
    normalised_features = scaler.transform(features).tolist()
    logging.info(str(normalised_features))
    return scaler, normalised_features


def train_models(song_data, test_song_data, features, test_features, listed_genres, test_listed_genres, vector_type):
    logging.info("--Normalisation...")
    scaler = MinMaxScaler()
    scaler.fit(features)
    normalised_features = scaler.transform(features).tolist()
    logging.info(str(normalised_features))

    # BEGIN STANDALONE SVM
    logging.info("--Training Classifier...")
    start = timer()
    clf = sklearn.svm.SVC().fit(normalised_features, listed_genres)
    end = timer()
    svm_time = end-start
    timing.info('Trained' + vector_type + 'SVM in ' + str(svm_time))

    normalised = normalised_features[::-1]
    predicted_genres = clf.predict(normalised).tolist()
    for song in song_data:
        if vector_type == "TIMBRE":
            song.normalised_timbre = normalised.pop()
            song.predicted_genre_timbre = predicted_genres.pop()
        elif vector_type == "MID":
            song.normalised_features = normalised.pop()
            song.predicted_genre_features = predicted_genres.pop()
        elif vector_type == "TIMBRE_SQ":
            song.normalised_timbre_sq = normalised.pop()
            song.predicted_genre_timbre_sq = predicted_genres.pop()
        elif vector_type == "MID_SQ":
            song.normalised_features_sq = normalised.pop()
            song.predicted_genre_features_sq = predicted_genres.pop()
        else:
            print("Invalid vector type selected")
            exit(1)

    # Testing
    logging.info("--Normalisation...")
    normalised_test_features = scaler.transform(test_features).tolist()
    logging.info(str(normalised_test_features))
    test_normalised = normalised_test_features[::-1]
    logging.info("--Prediction...")
    predicted_genres = clf.predict(test_normalised).tolist()
    predicted = predicted_genres[::-1]
    for test_song in test_song_data:
        if vector_type == "TIMBRE":
            test_song.normalised_timbre = test_normalised.pop()
            test_song.predicted_genre_timbre = predicted.pop()
        elif vector_type == "MID":
            test_song.normalised_features = test_normalised.pop()
            test_song.predicted_genre_features = predicted.pop()
        elif vector_type == "TIMBRE_SQ":
            test_song.normalised_timbre_sq = test_normalised.pop()
            test_song.predicted_genre_timbre_sq = predicted.pop()
        elif vector_type == "MID_SQ":
            test_song.normalised_features_sq = test_normalised.pop()
            test_song.predicted_genre_features_sq = predicted.pop()
        else:
            print("Invalid vector type selected")
            exit(1)

    logging.info("--Analysis...")
    non_matches = [(i, j) for i, j in zip(predicted_genres, test_listed_genres) if i != j]
    logging.info("Genre conflicts: " + str(non_matches))
    accuracy = len(non_matches) / len(predicted_genres)
    logging.info("Accuracy: " + str(accuracy) + "%")
    # END STANDALONE SVM

    # BEGIN STANDALONE K-MEANS
    genres = [song.listed_genre for song in song_data]
    start = timer()
    kmeans = sklearn.cluster.KMeans(len(set(genres))).fit(normalised_features)
    end = timer()
    timing.info('Trained' + vector_type + 'K-MEANS in ' + str(end - start))
    # END STANDALONE K-MEANS

    # BEGIN GENRE K-MEANS
    genre_kmeans = []
    start = timer()
    for cls in clf.classes_:
        if vector_type == "TIMBRE":
            cls_songs = [song for song in song_data if song.predicted_genre_timbre == cls]
        elif vector_type == "MID":
            cls_songs = [song for song in song_data if song.predicted_genre_features == cls]
        elif vector_type == "TIMBRE_SQ":
            cls_songs = [song for song in song_data if song.predicted_genre_timbre_sq == cls]
        elif vector_type == "MID_SQ":
            cls_songs = [song for song in song_data if song.predicted_genre_features_sq == cls]
        else:
            print("Invalid vector type selected")
            exit(1)
        if len(cls_songs) > 0:
            cls_features = []
            if vector_type == "TIMBRE":
                cls_features = [song.normalised_timbre for song in cls_songs]
            elif vector_type == "MID":
                cls_features = [song.normalised_features for song in cls_songs]
            elif vector_type == "TIMBRE_SQ":
                cls_features = [song.normalised_timbre_sq for song in cls_songs]
            elif vector_type == "MID_SQ":
                cls_features = [song.normalised_features_sq for song in cls_songs]
            else:
                print("Invalid vector type selected")
                exit(1)
            genre_kmeans.append((cls, sklearn.cluster.KMeans(min(10, len(cls_songs))).fit(cls_features)))
    end = timer()
    kmeans_genre_time = end - start
    total_time = svm_time + kmeans_genre_time
    timing.info('Trained' + vector_type + ' - KMEANS PER GENRE in ' + str(kmeans_genre_time) + ' Total time (inc. SVM for genre) ' + str(total_time))
    # END GENRE K-MEANS

    # BEGIN DBSCAN
    start = timer()
    dbscan = sklearn.cluster.DBSCAN().fit(normalised_features)
    end = timer()
    dbscan_time = end - start
    timing.info('Trained' + vector_type + 'DBSCAN in ' + str(dbscan_time))
    # END DBSCAN

    # BEGIN SVM ON DBSCAN todo: justify - simplify distance lookup/classification
    labels = dbscan.labels_.tolist()
    svm_on_dbscan = None
    if labels.count(labels[0]) != len(labels):
        start = timer()
        svm_on_dbscan = sklearn.svm.SVC().fit(normalised_features, dbscan.labels_)
        end = timer()
        svm_on_dbscan_time = end - start
        total_time = dbscan_time + svm_on_dbscan_time
        timing.info('Trained' + vector_type + ' - SVM ON DBSCAN in ' + str(svm_on_dbscan_time) + ' Total time (inc. DBSCAN) ' + str(total_time))

        for song in song_data:
            if vector_type == "TIMBRE":
                [song.dbscan_cluster_id_timbre] = svm_on_dbscan.predict([song.normalised_timbre])
            elif vector_type == "MID":
                [song.dbscan_cluster_id_features] = svm_on_dbscan.predict([song.normalised_features])
            elif vector_type == "TIMBRE_SQ":
                [song.dbscan_cluster_id_timbre_sq] = svm_on_dbscan.predict([song.normalised_timbre_sq])
            elif vector_type == "MID_SQ":
                [song.dbscan_cluster_id_features_sq] = svm_on_dbscan.predict([song.normalised_features_sq])
            else:
                print("Invalid vector type selected")
                exit(1)
        for test_song in test_song_data:
            if vector_type == "TIMBRE":
                [test_song.dbscan_cluster_id_timbre] = svm_on_dbscan.predict([test_song.normalised_timbre])
            elif vector_type == "MID":
                [test_song.dbscan_cluster_id_features] = svm_on_dbscan.predict([test_song.normalised_features])
            elif vector_type == "TIMBRE_SQ":
                [test_song.dbscan_cluster_id_timbre_sq] = svm_on_dbscan.predict([test_song.normalised_timbre_sq])
            elif vector_type == "MID_SQ":
                [test_song.dbscan_cluster_id_features_sq] = svm_on_dbscan.predict([test_song.normalised_features_sq])
            else:
                print("Invalid vector type selected")
                exit(1)
    # END SVM ON DBSCAN

    # BEGIN GENRE DBSCAN
    genre_dbscan = []
    start = timer()
    for cls in clf.classes_:
        if vector_type == "TIMBRE":
            cls_songs = [song for song in song_data if song.predicted_genre_timbre == cls]
        elif vector_type == "MID":
            cls_songs = [song for song in song_data if song.predicted_genre_features == cls]
        elif vector_type == "TIMBRE_SQ":
            cls_songs = [song for song in song_data if song.predicted_genre_timbre_sq == cls]
        elif vector_type == "MID_SQ":
            cls_songs = [song for song in song_data if song.predicted_genre_features_sq == cls]
        else:
            print("Invalid vector type selected")
            exit(1)
        if len(cls_songs) > 0:
            cls_features = []
            if vector_type == "TIMBRE":
                cls_features = [song.normalised_timbre for song in cls_songs]
            elif vector_type == "MID":
                cls_features = [song.normalised_features for song in cls_songs]
            elif vector_type == "TIMBRE_SQ":
                cls_features = [song.normalised_timbre_sq for song in cls_songs]
            elif vector_type == "MID_SQ":
                cls_features = [song.normalised_features_sq for song in cls_songs]
            else:
                print("Invalid vector type selected")
                exit(1)
            genre_dbscan.append((cls, sklearn.cluster.DBSCAN().fit(cls_features)))
    end = timer()
    genre_dbscan_time = end - start
    total_time = svm_time + genre_dbscan_time
    timing.info('Trained ' + vector_type + ' - DBSCAN PER GENRE in ' + str(genre_dbscan_time) + ' Total time (inc. SVM for genre) ' + str(total_time))
    # END GENRE DBSCAN

    logging.info("--Storage...")
    # Song Data
    joblib.dump(scaler, 'data/scaler_' + vector_type.lower() + '.pkl')
    # Classifiers & Clusters
    joblib.dump(clf, 'data/classifier_' + vector_type.lower() + '.pkl')
    joblib.dump(kmeans, 'data/kmeans_' + vector_type.lower() + '.pkl')
    joblib.dump(genre_kmeans, 'data/genre_kmeans_' + vector_type.lower() + '.pkl')
    joblib.dump(dbscan, 'data/dbscan_' + vector_type.lower() + '.pkl')
    if svm_on_dbscan is not None:
        joblib.dump(svm_on_dbscan, 'data/svm_on_dbscan_' + vector_type.lower() + '.pkl')
    joblib.dump(genre_dbscan, 'data/genre_dbscan_' + vector_type.lower() + '.pkl')


def create():
    # ------------------
    # General Setup/Song Operations
    # ------------------
    logging.info("Starting setup...")
    logging.info("-Starting Training...")

    logging.info("--File Conversion and Metadata Retrieval (inc. features)...")
    song_data = calculate_or_load("data/song_data.pkl", convert_and_get_data, paths.training_src_path, paths.training_dst_path)

    logging.info("--Feature to List...")
    mid_features, timbre_features, mid_sq_features, timbre_sq_features, listed_genres = songs_to_features(song_data)

    logging.info("-Starting Testing...")
    logging.info("--File Conversion and Metadata Retrieval (inc. features)...")
    test_song_data = calculate_or_load("data/test_song_data.pkl", convert_and_get_data, paths.test_src_path, paths.test_dst_path)

    logging.info("--Feature to List...")
    test_mid_features, test_timbre_features, test_mid_sq_features, test_timbre_sq_features, test_listed_genres = songs_to_features(test_song_data)

    train_models(song_data, test_song_data, timbre_features, test_timbre_features, listed_genres, test_listed_genres, "TIMBRE")
    train_models(song_data, test_song_data, mid_features, test_mid_features, listed_genres, test_listed_genres, "MID")
    train_models(song_data, test_song_data, timbre_sq_features, test_timbre_sq_features, listed_genres, test_listed_genres, "TIMBRE_SQ")
    train_models(song_data, test_song_data, mid_sq_features, test_mid_sq_features, listed_genres, test_listed_genres, "MID_SQ")

    joblib.dump(song_data, "data/song_data.pkl")
    joblib.dump(test_song_data, "data/test_song_data.pkl")


if __name__ == "__main__":
    create()
