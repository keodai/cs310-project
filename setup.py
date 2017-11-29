import utils
import converter
import paths
from song import Song

import os
import logging
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
import sklearn


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


def classify():
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
    normalised_features = scaler.transform(features)
    logging.info(str(normalised_features))
    normalised = normalised_features[::-1]
    for song in song_data:
        song.normalised_features = normalised.pop()

    logging.info("--Training Classifier...")
    clf = sklearn.svm.SVC()
    clf.fit(normalised_features, listed_genres)

    logging.info("-Starting Testing...")
    logging.info("--File Conversion and Metadata Retrieval (inc. features)...")
    test_song_data = calculate_or_load("data/test_song_data.pkl",
                                       convert_and_get_data, paths.test_src_path, paths.test_dst_path)

    logging.info("--Feature to List...")
    test_features, test_listed_genres = songs_to_features(test_song_data)

    logging.info("--Normalisation...")
    normalised_test_features = scaler.transform(test_features)
    logging.info(str(normalised_test_features))
    test_normalised = normalised_test_features.reverse()
    for song in song_data:
        song.normalised_features = test_normalised.pop()

    logging.info("--Prediction...")
    predicted_genres = clf.predict(normalised_test_features)
    predicted = predicted_genres.reverse()
    for song in test_song_data:
        song.predicted_genre = predicted.pop()

    logging.info("--Analysis...")
    non_matches = [(i, j) for i, j in zip(predicted_genres, test_listed_genres) if i != j]
    logging.info("Genre conflicts: " + str(non_matches))
    accuracy = len(non_matches) / len(predicted_genres)
    logging.info("Accuracy: " + str(accuracy) + "%")

    logging.info("--Storage...")
    joblib.dump(song_data, "data/song_data.pkl")
    joblib.dump(scaler, "data/scaler.pkl")
    joblib.dump(clf, 'data/classifier.pkl')
    joblib.dump(test_song_data, "data/test_song_data.pkl")
