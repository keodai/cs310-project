import os
from sklearn.externals import joblib

# List files required for all forms of recommendation, created during setup
REQUIRED_FILES = {"song_data": "data/song_data.pkl",
                  "test_song_data": "data/test_song_data.pkl",
                  "scaler_timbre": "data/scaler_timbre.pkl",
                  "classifier_timbre": "data/classifier_timbre.pkl",
                  "kmeans_timbre": "data/kmeans_timbre.pkl",
                  "genre_kmeans_timbre": "data/genre_kmeans_timbre.pkl",
                  "dbscan_timbre": "data/dbscan_timbre.pkl",
                  "svm_on_dbscan_timbre": "data/svm_on_dbscan_timbre.pkl",
                  "genre_dbscan_timbre": "data/genre_dbscan_timbre.pkl",
                  "scaler_mid": "data/scaler_mid.pkl",
                  "classifier_mid": "data/classifier_mid.pkl",
                  "kmeans_mid": "data/kmeans_mid.pkl",
                  "genre_kmeans_mid": "data/genre_kmeans_mid.pkl",
                  "dbscan_mid": "data/dbscan_mid.pkl",
                  "svm_on_dbscan_mid": "data/svm_on_dbscan_mid.pkl",
                  "genre_dbscan_mid": "data/genre_dbscan_mid.pkl",
                  "scaler_timbre_sq": "data/scaler_timbre_sq.pkl",
                  "classifier_timbre_sq": "data/classifier_timbre_sq.pkl",
                  "kmeans_timbre_sq": "data/kmeans_timbre_sq.pkl",
                  "genre_kmeans_timbre_sq": "data/genre_kmeans_timbre_sq.pkl",
                  "dbscan_timbre_sq": "data/dbscan_timbre_sq.pkl",
                  "svm_on_dbscan_timbre_sq": "data/svm_on_dbscan_timbre_sq.pkl",
                  "genre_dbscan_timbre_sq": "data/genre_dbscan_timbre_sq.pkl",
                  "scaler_mid_sq": "data/scaler_mid_sq.pkl",
                  "classifier_mid_sq": "data/classifier_mid_sq.pkl",
                  "kmeans_mid_sq": "data/kmeans_mid_sq.pkl",
                  "genre_kmeans_mid_sq": "data/genre_kmeans_mid_sq.pkl",
                  "dbscan_mid_sq": "data/dbscan_mid_sq.pkl",
                  "svm_on_dbscan_mid_sq": "data/svm_on_dbscan_mid_sq.pkl",
                  "genre_dbscan_mid_sq": "data/genre_dbscan_mid_sq.pkl"}


# def select_data(item, vector_type, condition=True):
#     if condition:
#         name = item + '_' + vector_type.lower()
#         return joblib.load(REQUIRED_FILES[name])
#     else:
#         return None


def init():
    # Ensure all files have been created by setup
    required_files_present = [os.path.isfile(value) for value in REQUIRED_FILES.values()]
    if not all(required_files_present):
        raise IOError('Required data files or models are not present')
    print("Loading song data...")
    # Load required files for the current recommendation task
    song_data = joblib.load(REQUIRED_FILES['song_data'])
    print("Loading test song data...")
    test_song_data = joblib.load(REQUIRED_FILES['test_song_data'])

    # scaler = select_data('scaler', vector_type)
    # svm_classifier = select_data('classifier', vector_type)
    # kmeans = select_data('kmeans', vector_type, mode == "FASTKMEANS" or mode == "FASTSORTEDKMEANS" or mode == "KMEANS")
    # genre_kmeans = select_data('genre_kmeans', vector_type, mode == "SVM+KMEANS")
    # dbscan = select_data('dbscan', vector_type, mode == "DBSCAN" or mode == "DBSCAN+SVM")
    # svm_on_dbscan = select_data('svm_on_dbscan', vector_type, mode == "DBSCAN+SVM")
    # genre_dbscan = select_data('genre_dbscan', vector_type, mode == "SVM+DBSCAN")
    print("Loading scalers and ml models...")
    scaler_timbre = joblib.load(REQUIRED_FILES['scaler_timbre'])
    classifier_timbre = joblib.load(REQUIRED_FILES['classifier_timbre'])
    kmeans_timbre = joblib.load(REQUIRED_FILES['kmeans_timbre'])
    genre_kmeans_timbre = joblib.load(REQUIRED_FILES['genre_kmeans_timbre'])
    dbscan_timbre = joblib.load(REQUIRED_FILES['dbscan_timbre'])
    svm_on_dbscan_timbre = joblib.load(REQUIRED_FILES['svm_on_dbscan_timbre'])
    genre_dbscan_timbre = joblib.load(REQUIRED_FILES['genre_dbscan_timbre'])
    scaler_mid = joblib.load(REQUIRED_FILES['scaler_mid'])
    classifier_mid = joblib.load(REQUIRED_FILES['classifier_mid'])
    kmeans_mid = joblib.load(REQUIRED_FILES['kmeans_mid'])
    genre_kmeans_mid = joblib.load(REQUIRED_FILES['genre_kmeans_mid'])
    dbscan_mid = joblib.load(REQUIRED_FILES['dbscan_mid'])
    svm_on_dbscan_mid = joblib.load(REQUIRED_FILES['svm_on_dbscan_mid'])
    genre_dbscan_mid = joblib.load(REQUIRED_FILES['genre_dbscan_mid'])
    scaler_timbre_sq = joblib.load(REQUIRED_FILES['scaler_timbre_sq'])
    classifier_timbre_sq = joblib.load(REQUIRED_FILES['classifier_timbre_sq'])
    kmeans_timbre_sq = joblib.load(REQUIRED_FILES['kmeans_timbre_sq'])
    genre_kmeans_timbre_sq = joblib.load(REQUIRED_FILES['genre_kmeans_timbre_sq'])
    dbscan_timbre_sq = joblib.load(REQUIRED_FILES['dbscan_timbre_sq'])
    svm_on_dbscan_timbre_sq = joblib.load(REQUIRED_FILES['svm_on_dbscan_timbre_sq'])
    genre_dbscan_timbre_sq = joblib.load(REQUIRED_FILES['genre_dbscan_timbre_sq'])
    scaler_mid_sq = joblib.load(REQUIRED_FILES['scaler_mid_sq'])
    classifier_mid_sq = joblib.load(REQUIRED_FILES['classifier_mid_sq'])
    kmeans_mid_sq = joblib.load(REQUIRED_FILES['kmeans_mid_sq'])
    genre_kmeans_mid_sq = joblib.load(REQUIRED_FILES['genre_kmeans_mid_sq'])
    dbscan_mid_sq = joblib.load(REQUIRED_FILES['dbscan_mid_sq'])
    svm_on_dbscan_mid_sq = joblib.load(REQUIRED_FILES['svm_on_dbscan_mid_sq'])
    genre_dbscan_mid_sq = joblib.load(REQUIRED_FILES['genre_dbscan_mid_sq'])
    print("Loading complete")

    return {"song_data": song_data,
            "test_song_data": test_song_data,
            "scaler_timbre": scaler_timbre,
            "classifier_timbre": classifier_timbre,
            "kmeans_timbre": kmeans_timbre,
            "genre_kmeans_timbre": genre_kmeans_timbre,
            "dbscan_timbre": dbscan_timbre,
            "svm_on_dbscan_timbre": svm_on_dbscan_timbre,
            "genre_dbscan_timbre": genre_dbscan_timbre,
            "scaler_mid": scaler_mid,
            "classifier_mid": classifier_mid,
            "kmeans_mid": kmeans_mid,
            "genre_kmeans_mid": genre_kmeans_mid,
            "dbscan_mid": dbscan_mid,
            "svm_on_dbscan_mid": svm_on_dbscan_mid,
            "genre_dbscan_mid": genre_dbscan_mid,
            "scaler_timbre_sq": scaler_timbre_sq,
            "classifier_timbre_sq": classifier_timbre_sq,
            "kmeans_timbre_sq": kmeans_timbre_sq,
            "genre_kmeans_timbre_sq": genre_kmeans_timbre_sq,
            "dbscan_timbre_sq":dbscan_timbre_sq,
            "svm_on_dbscan_timbre_sq": svm_on_dbscan_timbre_sq,
            "genre_dbscan_timbre_sq": genre_dbscan_timbre_sq,
            "scaler_mid_sq": scaler_mid_sq,
            "classifier_mid_sq": classifier_mid_sq,
            "kmeans_mid_sq": kmeans_mid_sq,
            "genre_kmeans_mid_sq": genre_kmeans_mid_sq,
            "dbscan_mid_sq": dbscan_mid_sq,
            "svm_on_dbscan_mid_sq": svm_on_dbscan_mid_sq,
            "genre_dbscan_mid_sq": genre_dbscan_mid_sq}