import logging
import os
import shutil
import subprocess
from timeit import default_timer as timer

import numpy as np
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import multi_logging
import paths
from song import Song

# Loggers
timing = multi_logging.setup_logger('timing', 'logs/training_times.log')
cv = multi_logging.setup_logger('cv', 'logs/cv.log')


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
    mid_features = np.array([song.features for song in song_data])
    timbre_features = np.array([song.timbre for song in song_data])
    mid_sq_features = np.array([song.features_sq for song in song_data])
    timbre_sq_features = np.array([song.timbre_sq for song in song_data])
    short_timbre_features = np.array([song.short_timbre for song in song_data])
    short_mid_features = np.array([song.short_features for song in song_data])
    listed_genres = np.array([song.listed_genre for song in song_data])
    return mid_features, timbre_features, mid_sq_features, timbre_sq_features, short_timbre_features, short_mid_features, listed_genres


# Perform feature scaling to standardise attributes
def scale_data(features, scaler=None):
    # todo: should this be standardisation? MinMaxScaler
    if scaler is None:
        scaler = StandardScaler()
        normalised_features = scaler.fit_transform(features)
    else:
        normalised_features = scaler.transform(features)
    return scaler, normalised_features


# Get metric score for model by k-fold cross-validation on the given data
def cross_validation(model, desc, x, y, scoring=None, folds=10):
    scores = cross_val_score(model, x, y, cv=folds, scoring=scoring)
    cv.info("%s: %0.2f (+/- %0.2f)" % (desc, scores.mean(), scores.std()))
    cv.info(str(scores))


def timed_fit(clf, vector_type, model_desc, x, y=None):
    start = timer()
    clf = clf.fit(x, y)
    end = timer()
    calc_time = end - start
    timing.info('Trained ' + vector_type + ' ' + model_desc + ' in ' + str(calc_time) + 's')
    return clf


# Perform training of models using features of the given vector type
def train_models(song_data, test_song_data, features, test_features, listed_genres, test_listed_genres, vector_type):
    logging.info("-Feature scaling...")
    scaler, normalised_features = scale_data(features)

    print("-DT gini feature importance...")
    cv.info("DT gini feature importance:")
    dt = DecisionTreeClassifier().fit(normalised_features, listed_genres)
    importance = dt.feature_importances_
    cv.info(importance)

    # BEGIN STANDALONE SVM
    print("-SVM cross-validation...")
    cv.info("SVM cross-validation:")
    clf = SVC()
    cross_validation(clf, "SVM accuracy", normalised_features, listed_genres)
    print("-Training SVM...")
    clf = timed_fit(clf, vector_type, "SVM", normalised_features, listed_genres)

    print("-Linear SVM cross-validation...")
    cv.info("Linear SVM cross-validation:")
    clf_linear = SVC(kernel='linear')
    cross_validation(clf_linear, "Linear SVM accuracy", normalised_features, listed_genres)
    print("-Training Linear SVM...")
    clf_linear = timed_fit(clf_linear, vector_type, "Linear SVM", normalised_features, listed_genres)

    print("-Updating song data with predictions...")
    predicted_genres = clf.predict(normalised_features)
    for idx, song in enumerate(song_data):
        song.set_normalised_features(vector_type, normalised_features[idx])
        song.set_predicted_genre(vector_type, predicted_genres[idx])
    # END STANDALONE SVM

    # BEGIN KNN
    print("-KNN99 cross-validation...")
    cv.info("KNN99 cross-validation:")
    # Cross-validation cannot be performed on this model for a small dataset with n_samples < n_neighbors
    knn = KNeighborsClassifier(n_neighbors=99)
    cross_validation(knn, "KNN99 accuracy", normalised_features, listed_genres)
    print("-Training KNN99...")
    knn = timed_fit(knn, vector_type, "KNN99", normalised_features, listed_genres)

    print("-KNN11 cross-validation...")
    cv.info("KNN11 cross-validation:")
    knn11 = KNeighborsClassifier(n_neighbors=11)
    cross_validation(knn11, "KNN11 accuracy", normalised_features, listed_genres)
    print("-Training KNN11...")
    knn11 = timed_fit(knn11, vector_type, "KNN11", normalised_features, listed_genres)
    # END KNN

    # CLASSIFIER TESTING
    if np.size(test_features) > 0:
        print("-Test feature scaling...")
        normalised_test_features = scaler.transform(test_features)
        print("-Updating test song data with predictions...")
        test_predicted_genres = clf.predict(normalised_test_features)
        for idx, test_song in enumerate(test_song_data):
            test_song.set_normalised_features(vector_type, normalised_test_features[idx])
            test_song.set_predicted_genre(vector_type, test_predicted_genres[idx])

        print("-Testing SVM...")
        accuracy = clf.score(normalised_test_features, test_listed_genres)
        cv.info("SVM Test Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std()))

        print("-Testing Linear SVM...")
        accuracy = clf_linear.score(normalised_test_features, test_listed_genres)
        cv.info("Linear SVM Test Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std()))

        print("-Testing KNN99...")
        accuracy = knn.score(normalised_test_features, test_listed_genres)
        cv.info("KNN99 Test Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std()))

        print("-Testing KNN11...")
        accuracy = knn11.score(normalised_test_features, test_listed_genres)
        cv.info("KNN11 Test Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(), accuracy.std()))
    # END CLASSIFIER TESTING

    # BEGIN STANDALONE K-MEANS
    genres = [song.listed_genre for song in song_data]
    print("-KMEANSFIXED cross-validation...")
    cv.info("KMEANSFIXED cross-validation:")
    kmeans_fixed = KMeans(15)
    cross_validation(kmeans_fixed, "KMEANSFIXED completeness", normalised_features, listed_genres, scoring='completeness_score')
    cross_validation(kmeans_fixed, "KMEANSFIXED homogeneity", normalised_features, listed_genres, scoring='homogeneity_score')
    cross_validation(kmeans_fixed, "KMEANSFIXED v-measure", normalised_features, listed_genres, scoring='v_measure_score')
    print("-Training KMEANSFIXED...")
    kmeans_fixed = timed_fit(kmeans_fixed, vector_type, "KMEANSFIXED", normalised_features)

    print("-KMEANS cross-validation...")
    cv.info("KMEANS cross-validation:")
    kmeans = KMeans(len(set(genres)))
    cross_validation(kmeans, "KMEANS completeness", normalised_features, listed_genres, scoring='completeness_score')
    cross_validation(kmeans, "KMEANS homogeneity", normalised_features, listed_genres, scoring='homogeneity_score')
    cross_validation(kmeans, "KMEANS v-measure", normalised_features, listed_genres, scoring='v_measure_score')
    print("-Training KMEANS...")
    kmeans = timed_fit(kmeans, vector_type, "KMEANS", normalised_features)

    print("-KMEANS2 cross-validation...")
    cv.info("KMEANS2 cross-validation:")
    kmeans2 = KMeans(2 * len(set(genres)))
    cross_validation(kmeans2, "KMEANS2 completeness", normalised_features, listed_genres, scoring='completeness_score')
    cross_validation(kmeans2, "KMEANS2 homogeneity", normalised_features, listed_genres, scoring='homogeneity_score')
    cross_validation(kmeans2, "KMEANS2 v-measure", normalised_features, listed_genres, scoring='v_measure_score')
    print("-Training KMEANS2...")
    kmeans2 = timed_fit(kmeans2, vector_type, "KMEANS2", normalised_features)

    print("-Testing KMEANS...")
    h, c, v = homogeneity_completeness_v_measure(listed_genres, kmeans.labels_)
    cv.info("KMEANS hcv: %0.2f, %0.2f, %0.2f" % (h, c, v))

    print("-Testing KMEANS2...")
    h, c, v = homogeneity_completeness_v_measure(listed_genres, kmeans2.labels_)
    cv.info("KMEANS2 hcv: %0.2f, %0.2f, %0.2f" % (h, c, v))

    # END STANDALONE K-MEANS

    # BEGIN STANDALONE DBSCAN
    print("-DBSCAN cross-validation...")
    cv.info("DBSCAN cross-validation:")
    dbscan = DBSCAN(eps=2.5, min_samples=5)
    print("-Training DBSCAN...")
    dbscan = timed_fit(dbscan, vector_type, "DBSCAN", normalised_features)

    print("-Testing DBSCAN...")
    h, c, v = homogeneity_completeness_v_measure(listed_genres, dbscan.labels_)
    cv.info("DBSCAN hcv: %0.2f, %0.2f, %0.2f" % (h, c, v))
    # END STANDALONE DBSCAN

    # # BEGIN GENRE K-MEANS
    # print("-Training GENRE_KMEANS...")
    # genre_kmeans = []
    # start = timer()
    # for cls in clf.classes_:
    #     cls_songs = [song for song in song_data if song.get_predicted_genre(vector_type) == cls]
    #     if len(cls_songs) > 0:
    #         cls_features = np.array([song.get_normalised_features(vector_type) for song in cls_songs])
    #         genre_kmeans.append((cls, KMeans(min(10, len(cls_songs))).fit(cls_features)))
    # end = timer()
    # kmeans_genre_time = end - start
    # timing.info('Trained ' + vector_type + ' - GENRE_KMEANS in an additional' + str(kmeans_genre_time))
    # # END GENRE K-MEANS
    #
    # # BEGIN GENRE DBSCAN
    # print("-Training GENRE_DBSCAN...")
    # genre_dbscan = []
    # start = timer()
    # for cls in clf.classes_:
    #     cls_songs = [song for song in song_data if song.get_predicted_genre(vector_type) == cls]
    #     if len(cls_songs) > 0:
    #         cls_features = np.array([song.get_normalised_features(vector_type) for song in cls_songs])
    #         genre_dbscan.append((cls, DBSCAN(eps=2.5, min_samples=5).fit(cls_features)))
    # end = timer()
    # genre_dbscan_time = end - start
    # timing.info('Trained ' + vector_type + ' - GENRE_DBSCAN in ' + str(genre_dbscan_time))
    # # END GENRE DBSCAN
    #
    # # BEGIN SVM ON DBSCAN
    # print("-Training SVM_ON_DBSCAN...")
    # labels = dbscan.labels_
    # # Need >1 labels to be able to train an SVM classifier
    # svm_on_dbscan = SVC()
    # if np.size(np.unique(labels)) > 1:
    #     svm_on_dbscan = timed_fit(svm_on_dbscan, vector_type, "SVM_ON_DBSCAN", normalised_features, dbscan.labels_)
    #
    #     for song in song_data:
    #         song.set_dbscan_cluster_id(vector_type, svm_on_dbscan.predict([song.get_normalised_features(vector_type)])[0])
    #     for test_song in test_song_data:
    #         test_song.set_dbscan_cluster_id(vector_type, svm_on_dbscan.predict([test_song.get_normalised_features(vector_type)])[0])
    # # END SVM ON DBSCAN
    #
    # print("-Storage...")
    # # Song Data
    # joblib.dump(scaler, 'data/scaler_' + vector_type.lower() + '.pkl')
    # # Classifiers & Clusters
    # joblib.dump(clf, 'data/classifier_' + vector_type.lower() + '.pkl')
    # joblib.dump(clf_linear, 'data/classifier_linear_' + vector_type.lower() + '.pkl')
    # joblib.dump(knn, 'data/knn_' + vector_type.lower() + '.pkl')
    # joblib.dump(knn11, 'data/knn11_' + vector_type.lower() + '.pkl')
    # joblib.dump(kmeans, 'data/kmeans_fixed_' + vector_type.lower() + '.pkl')
    # joblib.dump(kmeans, 'data/kmeans_' + vector_type.lower() + '.pkl')
    # joblib.dump(kmeans2, 'data/kmeans2_' + vector_type.lower() + '.pkl')
    # joblib.dump(dbscan, 'data/dbscan_' + vector_type.lower() + '.pkl')
    # joblib.dump(genre_kmeans, 'data/genre_kmeans_' + vector_type.lower() + '.pkl')
    # joblib.dump(genre_dbscan, 'data/genre_dbscan_' + vector_type.lower() + '.pkl')
    # if svm_on_dbscan is not None:
    #     joblib.dump(svm_on_dbscan, 'data/svm_on_dbscan_' + vector_type.lower() + '.pkl')


# Song data conversion/preprocessing and model training
def create():
    # General Setup/Song Operations
    print("Running training/setup...")
    print("Converting training files and extracting features...")
    song_data = calculate_or_load("data/song_data.pkl", convert_and_get_data, paths.training_src_path, paths.training_dst_path)
    print("Features to list...")
    # todo: convert these to pandas dataframe and perform manipulation
    mid_features, timbre_features, mid_sq_features, timbre_sq_features, short_timbre_features, short_mid_features, listed_genres = songs_to_features(song_data)

    print("Converting test files and extracting features...")
    test_song_data = calculate_or_load("data/test_song_data.pkl", convert_and_get_data, paths.test_src_path, paths.test_dst_path)
    logging.info("Features to List...")
    test_mid_features, test_timbre_features, test_mid_sq_features, test_timbre_sq_features, test_short_timbre_features, test_short_mid_features, test_listed_genres = songs_to_features(test_song_data)

    print("Writing training song data...")
    joblib.dump(song_data, "data/song_data.pkl")
    print("Writing test song data...")
    joblib.dump(test_song_data, "data/test_song_data.pkl")

    # Train models for each vector type
    print("Training models (TIMBRE)...")
    train_models(song_data, test_song_data, timbre_features, test_timbre_features, listed_genres, test_listed_genres, "TIMBRE")
    print("Training models (MID)...")
    train_models(song_data, test_song_data, mid_features, test_mid_features, listed_genres, test_listed_genres, "MID")
    print("Training models (TIMBRE_SQ)...")
    train_models(song_data, test_song_data, timbre_sq_features, test_timbre_sq_features, listed_genres, test_listed_genres, "TIMBRE_SQ")
    print("Training models (MID_SQ)...")
    train_models(song_data, test_song_data, mid_sq_features, test_mid_sq_features, listed_genres, test_listed_genres, "MID_SQ")
    print("Training models (SHORT_TIMBRE)...")
    train_models(song_data, test_song_data, short_timbre_features, test_short_timbre_features, listed_genres, test_listed_genres, "SHORT_TIMBRE")
    print("Training models (SHORT_MID)...")
    train_models(song_data, test_song_data, short_mid_features, test_short_mid_features, listed_genres, test_listed_genres, "SHORT_MID")

    print("Writing training song data...")
    joblib.dump(song_data, "data/song_data.pkl")
    print("Writing test song data...")
    joblib.dump(test_song_data, "data/test_song_data.pkl")


if __name__ == "__main__":
    create()
