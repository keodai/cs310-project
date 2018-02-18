import paths
from song import Song

import os
import subprocess
import logging
import multi_logging
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
import sklearn
from timeit import default_timer as timer
import shutil

# CONVERSION/NORMALISATION
# ------------------------
# SVM
# INDIVIDUAL K-MEANS
# GENRE K-MEANS
# DBSCAN
# DBSCAN ON SVM
# SVM ON DBSCAN

# Loggers
logging = multi_logging.setup_logger('output', 'logs/output.log')
timing = multi_logging.setup_logger('timing', 'logs/training_times.log')


# Return paths to non-hidden files in a directory, excluding sub-directories
def visible(src_path):
    return [os.path.join(src_path, file)
            for file in os.listdir(src_path)
            if not file.startswith('.') and os.path.isfile(os.path.join(src_path, file))]


# Convert input file (mp3) to wav
def convert(src, dst_dir, dst_ext):
    base = os.path.basename(src)
    dst = dst_dir + os.path.splitext(base)[0] + dst_ext
    try:
        subprocess.check_output(["ffmpeg", "-y", "-i", src, dst], stderr=subprocess.STDOUT)
        return dst
    except subprocess.CalledProcessError as e:
        logging.error(e)
        return


# Convert individual song to wav format, move to destination directory and create a Song object containing its data
def single_song(src, dst_path):
    logging.info("Processing: " + src)
    if src.endswith('.wav'):
        base = os.path.basename(src)
        dst = dst_path + os.path.splitext(base)[0] + paths.dst_ext
        shutil.copy2(src, dst)
    else:
        dst = convert(src, dst_path, paths.dst_ext)

    return Song(src, dst)


# Perform conversion and data/feature extraction on all songs in a directory
def convert_and_get_data(src_path, dst_path):
    song_data = []
    for src in visible(src_path):
        song_data.append(single_song(src, dst_path))
    return song_data


# Load song data if it already exists, otherwise convert songs and generate data objects again
def calculate_or_load(filename, fun, *args):
    if os.path.isfile(filename):
        return joblib.load(filename)
    else:
        return fun(*args)


# Retrieve feature information from a list of Song data objects
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
        song.set_normalised_features(vector_type, normalised.pop())
        song.set_predicted_genre(vector_type, predicted_genres.pop())

    # Testing
    logging.info("--Normalisation...")
    normalised_test_features = scaler.transform(test_features).tolist()
    logging.info(str(normalised_test_features))
    test_normalised = normalised_test_features[::-1]
    logging.info("--Prediction...")
    test_predicted_genres = clf.predict(test_normalised).tolist()
    for test_song in test_song_data:
        test_song.set_normalised_features(vector_type, test_normalised.pop())
        test_song.set_predicted_genre(vector_type, test_predicted_genres.pop())

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
        cls_songs = [song for song in song_data if song.get_predicted_genre(vector_type) == cls]
        if len(cls_songs) > 0:
            cls_features = [song.get_normalised_features(vector_type) for song in cls_songs]
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
            song.set_dbscan_cluster_id(vector_type, svm_on_dbscan.predict([song.get_normalised_features(vector_type)])[0])
        for test_song in test_song_data:
            test_song.set_dbscan_cluster_id(vector_type, svm_on_dbscan.predict([test_song.get_normalised_features(vector_type)])[0])
    # END SVM ON DBSCAN

    # BEGIN GENRE DBSCAN
    genre_dbscan = []
    start = timer()
    for cls in clf.classes_:
        cls_songs = [song for song in song_data if song.get_predicted_genre(vector_type) == cls]
        if len(cls_songs) > 0:
            cls_features = [song.get_normalised_features(vector_type) for song in cls_songs]
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


# Song data conversion/preprocessing and model training
def create():
    # General Setup/Song Operations
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

    # Train models for each vector type
    train_models(song_data, test_song_data, timbre_features, test_timbre_features, listed_genres, test_listed_genres, "TIMBRE")
    train_models(song_data, test_song_data, mid_features, test_mid_features, listed_genres, test_listed_genres, "MID")
    train_models(song_data, test_song_data, timbre_sq_features, test_timbre_sq_features, listed_genres, test_listed_genres, "TIMBRE_SQ")
    train_models(song_data, test_song_data, mid_sq_features, test_mid_sq_features, listed_genres, test_listed_genres, "MID_SQ")

    joblib.dump(song_data, "data/song_data.pkl")
    joblib.dump(test_song_data, "data/test_song_data.pkl")


if __name__ == "__main__":
    create()
