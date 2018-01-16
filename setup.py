import utils
import converter
import paths
from song import Song

import os
import logging
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
import sklearn


# CONVERSION/NORMALISATION
# ------------------------
# SVM
# INDIVIDUAL K-MEANS
# GENRE K-MEANS
# DBSCAN
# DBSCAN ON SVM
# SVM ON DBSCAN


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
    features = []
    listed_genres = []
    for song in song_data:
        features.append(song.features)
        listed_genres.append(song.listed_genre)
    return features, listed_genres


def db_cluster(data):
    return sklearn.cluster.DBSCAN().fit(data)


def create():
    logging.basicConfig(filename="logs/output.log", level=logging.DEBUG, format="%(asctime)s %(message)s")
    logging.info("Starting setup...")
    logging.info("-Starting Training...")

    logging.info("--File Conversion and Metadata Retrieval (inc. features)...")
    song_data = calculate_or_load("data/song_data.pkl",
                                  convert_and_get_data, paths.training_src_path, paths.training_dst_path)

    logging.info("--Feature to List...")
    features, listed_genres = songs_to_features(song_data)

    logging.info("--Normalisation...")
    scaler = MinMaxScaler()
    scaler.fit(features)
    normalised_features = scaler.transform(features).tolist()
    logging.info(str(normalised_features))

    # BEGIN STANDALONE SVM
    logging.info("--Training Classifier...")
    clf = sklearn.svm.SVC().fit(normalised_features, listed_genres)

    normalised = normalised_features[::-1]
    predicted_genres = clf.predict(normalised).tolist()
    for song in song_data:
        song.normalised_features = normalised.pop()
        song.predicted_genre = predicted_genres.pop()

    logging.info("-Starting Testing...")
    logging.info("--File Conversion and Metadata Retrieval (inc. features)...")
    test_song_data = calculate_or_load("data/test_song_data.pkl",
                                       convert_and_get_data, paths.test_src_path, paths.test_dst_path)

    logging.info("--Feature to List...")
    test_features, test_listed_genres = songs_to_features(test_song_data)

    logging.info("--Normalisation...")
    normalised_test_features = scaler.transform(test_features).tolist()
    logging.info(str(normalised_test_features))
    test_normalised = normalised_test_features[::-1]
    for test_song in test_song_data:
        test_song.normalised_features = test_normalised.pop()

    logging.info("--Prediction...")
    predicted_genres = clf.predict(normalised_test_features).tolist()
    predicted = predicted_genres[::-1]
    for test_song in test_song_data:
        test_song.predicted_genre = predicted.pop()

    logging.info("--Analysis...")
    non_matches = [(i, j) for i, j in zip(predicted_genres, test_listed_genres) if i != j]
    logging.info("Genre conflicts: " + str(non_matches))
    accuracy = len(non_matches) / len(predicted_genres)
    logging.info("Accuracy: " + str(accuracy) + "%")
    # END STANDALONE SVM

    # BEGIN STANDALONE K-MEANS
    genres = [song.listed_genre for song in song_data]
    kmeans = sklearn.cluster.KMeans(len(set(genres))).fit(normalised_features)
    # END STANDALONE K-MEANS

    # BEGIN GENRE K-MEANS
    genre_kmeans = []
    for cls in clf.classes_:
        cls_songs = [song for song in song_data if song.predicted_genre == cls]
        if len(cls_songs) > 0:
            cls_features = [song.normalised_features for song in cls_songs]
            genre_kmeans.append((cls, sklearn.cluster.KMeans(min(10, len(cls_songs))).fit(cls_features)))
    # END GENRE K-MEANS

    # BEGIN DBSCAN
    dbscan = sklearn.cluster.DBSCAN().fit(normalised_features)
    # END DBSCAN

    # BEGIN SVM ON DBSCAN todo: justify
    svm_on_dbscan = sklearn.svm.SVC().fit(normalised_features, dbscan.labels_)
    for song in song_data:
        [song.dbscan_cluster_id] = svm_on_dbscan.predict([song.normalised_features])
    for test_song in test_song_data:
        [test_song.dbscan_cluster_id] = svm_on_dbscan.predict([test_song.normalised_features])
    # END SVM ON DBSCAN

    # BEGIN GENRE DBSCAN
    genre_dbscan = []
    for cls in clf.classes_:
        cls_songs = [song for song in song_data if song.predicted_genre == cls]
        if len(cls_songs) > 0:
            cls_features = [song.normalised_features for song in cls_songs]
            genre_dbscan.append((cls, sklearn.cluster.DBSCAN().fit(cls_features)))
    # END GENRE DBSCAN

    logging.info("--Storage...")
    # Song Data
    joblib.dump(song_data, "data/song_data.pkl")
    joblib.dump(test_song_data, "data/test_song_data.pkl")
    joblib.dump(scaler, "data/scaler.pkl")

    # Classifiers & Clusters
    joblib.dump(clf, 'data/classifier.pkl')
    joblib.dump(kmeans, 'data/kmeans.pkl')
    joblib.dump(genre_kmeans, 'data/genre_kmeans.pkl')
    joblib.dump(dbscan, 'data/dbscan.pkl')
    joblib.dump(svm_on_dbscan, 'data/svm_on_dbscan.pkl')
    joblib.dump(genre_dbscan, 'data/genre_dbscan.pkl')

    # ------------------
    # Mid-level features
    # ------------------


if __name__ == "__main__":
    create()