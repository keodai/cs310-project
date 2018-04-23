# Checks all files that may be required for recommendation exist and loads them from disk into dictionary
# Performed on server start to speed up recommendation process

import os
from sklearn.externals import joblib

# List files required for all forms of recommendation, created during setup
REQUIRED_FILES = {"song_data": "data/song_data.pkl",
                  "test_song_data": "data/test_song_data.pkl",

                  "scaler_timbre": "data/scaler_timbre.pkl",
                  "classifier_timbre": "data/classifier_timbre.pkl",
                  "classifier_linear_timbre": "data/classifier_linear_timbre.pkl",
                  "knn_timbre": "data/knn_timbre.pkl",
                  "knn11_timbre": "data/knn11_timbre.pkl",
                  "kmeans_fixed_timbre": "data/kmeans_fixed_timbre.pkl",
                  "kmeans_timbre": "data/kmeans_timbre.pkl",
                  "kmeans2_timbre": "data/kmeans2_timbre.pkl",
                  "dbscan_timbre": "data/dbscan_timbre.pkl",
                  "genre_kmeans_timbre": "data/genre_kmeans_timbre.pkl",
                  "genre_dbscan_timbre": "data/genre_dbscan_timbre.pkl",
                  "svm_on_dbscan_timbre": "data/svm_on_dbscan_timbre.pkl",
                  
                  "scaler_mid": "data/scaler_mid.pkl",
                  "classifier_mid": "data/classifier_mid.pkl",
                  "classifier_linear_mid": "data/classifier_linear_mid.pkl",
                  "knn_mid": "data/knn_mid.pkl",
                  "knn11_mid": "data/knn11_mid.pkl",
                  "kmeans_fixed_mid": "data/kmeans_fixed_mid.pkl",
                  "kmeans_mid": "data/kmeans_mid.pkl",
                  "kmeans2_mid": "data/kmeans2_mid.pkl",
                  "dbscan_mid": "data/dbscan_mid.pkl",
                  "genre_kmeans_mid": "data/genre_kmeans_mid.pkl",
                  "genre_dbscan_mid": "data/genre_dbscan_mid.pkl",
                  "svm_on_dbscan_mid": "data/svm_on_dbscan_mid.pkl",

                  "scaler_timbre_sq": "data/scaler_timbre_sq.pkl",
                  "classifier_timbre_sq": "data/classifier_timbre_sq.pkl",
                  "classifier_linear_timbre_sq": "data/classifier_linear_timbre_sq.pkl",
                  "knn_timbre_sq": "data/knn_timbre_sq.pkl",
                  "knn11_timbre_sq": "data/knn11_timbre_sq.pkl",
                  "kmeans_fixed_timbre_sq": "data/kmeans_fixed_timbre_sq.pkl",
                  "kmeans_timbre_sq": "data/kmeans_timbre_sq.pkl",
                  "kmeans2_timbre_sq": "data/kmeans2_timbre_sq.pkl",
                  "dbscan_timbre_sq": "data/dbscan_timbre_sq.pkl",
                  "genre_kmeans_timbre_sq": "data/genre_kmeans_timbre_sq.pkl",
                  "genre_dbscan_timbre_sq": "data/genre_dbscan_timbre_sq.pkl",
                  "svm_on_dbscan_timbre_sq": "data/svm_on_dbscan_timbre_sq.pkl",

                  "scaler_mid_sq": "data/scaler_mid_sq.pkl",
                  "classifier_mid_sq": "data/classifier_mid_sq.pkl",
                  "classifier_linear_mid_sq": "data/classifier_linear_mid_sq.pkl",
                  "knn_mid_sq": "data/knn_mid_sq.pkl",
                  "knn11_mid_sq": "data/knn11_mid_sq.pkl",
                  "kmeans_fixed_mid_sq": "data/kmeans_fixed_mid_sq.pkl",
                  "kmeans_mid_sq": "data/kmeans_mid_sq.pkl",
                  "kmeans2_mid_sq": "data/kmeans2_mid_sq.pkl",
                  "dbscan_mid_sq": "data/dbscan_mid_sq.pkl",
                  "genre_kmeans_mid_sq": "data/genre_kmeans_mid_sq.pkl",
                  "genre_dbscan_mid_sq": "data/genre_dbscan_mid_sq.pkl",
                  "svm_on_dbscan_mid_sq": "data/svm_on_dbscan_mid_sq.pkl",

                  "scaler_short_timbre": "data/scaler_short_timbre.pkl",
                  "classifier_short_timbre": "data/classifier_short_timbre.pkl",
                  "classifier_linear_short_timbre": "data/classifier_linear_short_timbre.pkl",
                  "knn_short_timbre": "data/knn_short_timbre.pkl",
                  "knn11_short_timbre": "data/knn11_short_timbre.pkl",
                  "kmeans_fixed_short_timbre": "data/kmeans_fixed_short_timbre.pkl",
                  "kmeans_short_timbre": "data/kmeans_short_timbre.pkl",
                  "kmeans2_short_timbre": "data/kmeans2_short_timbre.pkl",
                  "dbscan_short_timbre": "data/dbscan_short_timbre.pkl",
                  "genre_kmeans_short_timbre": "data/genre_kmeans_short_timbre.pkl",
                  "genre_dbscan_short_timbre": "data/genre_dbscan_short_timbre.pkl",
                  "svm_on_dbscan_short_timbre": "data/svm_on_dbscan_short_timbre.pkl",

                  "scaler_short_mid": "data/scaler_short_mid.pkl",
                  "classifier_short_mid": "data/classifier_short_mid.pkl",
                  "classifier_linear_short_mid": "data/classifier_linear_short_mid.pkl",
                  "knn_short_mid": "data/knn_short_mid.pkl",
                  "knn11_short_mid": "data/knn11_short_mid.pkl",
                  "kmeans_fixed_short_mid": "data/kmeans_fixed_short_mid.pkl",
                  "kmeans_short_mid": "data/kmeans_short_mid.pkl",
                  "kmeans2_short_mid": "data/kmeans2_short_mid.pkl",
                  "dbscan_short_mid": "data/dbscan_short_mid.pkl",
                  "genre_kmeans_short_mid": "data/genre_kmeans_short_mid.pkl",
                  "genre_dbscan_short_mid": "data/genre_dbscan_short_mid.pkl",
                  "svm_on_dbscan_short_mid": "data/svm_on_dbscan_short_mid.pkl"}


# Load all files from disk that may be required to perform recommendation
# (on server start to speed up recommendation process)
def init():
    # Ensure all files have been created by setup
    required_files_present = [os.path.isfile(value) for value in REQUIRED_FILES.values()]
    if not all(required_files_present):
        raise IOError('Required data files or models are not present\n' + str(zip(REQUIRED_FILES.keys(),required_files_present)))

    # Load song data from disk
    print("Loading song data...")
    song_data = joblib.load(REQUIRED_FILES['song_data'])
    print("Loading test song data...")
    test_song_data = joblib.load(REQUIRED_FILES['test_song_data'])
    print("Loading scalers and ml models...")
    scaler_timbre = joblib.load(REQUIRED_FILES['scaler_timbre'])
    classifier_timbre = joblib.load(REQUIRED_FILES['classifier_timbre'])
    classifier_linear_timbre = joblib.load(REQUIRED_FILES['classifier_linear_timbre'])
    knn_timbre = joblib.load(REQUIRED_FILES['knn_timbre'])
    knn11_timbre = joblib.load(REQUIRED_FILES['knn11_timbre'])
    kmeans_fixed_timbre = joblib.load(REQUIRED_FILES['kmeans_fixed_timbre'])
    kmeans_timbre = joblib.load(REQUIRED_FILES['kmeans_timbre'])
    kmeans2_timbre = joblib.load(REQUIRED_FILES['kmeans2_timbre'])
    dbscan_timbre = joblib.load(REQUIRED_FILES['dbscan_timbre'])
    genre_kmeans_timbre = joblib.load(REQUIRED_FILES['genre_kmeans_timbre'])
    genre_dbscan_timbre = joblib.load(REQUIRED_FILES['genre_dbscan_timbre'])
    svm_on_dbscan_timbre = joblib.load(REQUIRED_FILES['svm_on_dbscan_timbre'])

    scaler_mid = joblib.load(REQUIRED_FILES['scaler_mid'])
    classifier_mid = joblib.load(REQUIRED_FILES['classifier_mid'])
    classifier_linear_mid = joblib.load(REQUIRED_FILES['classifier_linear_mid'])
    knn_mid = joblib.load(REQUIRED_FILES['knn_mid'])
    knn11_mid = joblib.load(REQUIRED_FILES['knn11_mid'])
    kmeans_fixed_mid = joblib.load(REQUIRED_FILES['kmeans_fixed_mid'])
    kmeans_mid = joblib.load(REQUIRED_FILES['kmeans_mid'])
    kmeans2_mid = joblib.load(REQUIRED_FILES['kmeans2_mid'])
    dbscan_mid = joblib.load(REQUIRED_FILES['dbscan_mid'])
    genre_kmeans_mid = joblib.load(REQUIRED_FILES['genre_kmeans_mid'])
    genre_dbscan_mid = joblib.load(REQUIRED_FILES['genre_dbscan_mid'])
    svm_on_dbscan_mid = joblib.load(REQUIRED_FILES['svm_on_dbscan_mid'])

    scaler_timbre_sq = joblib.load(REQUIRED_FILES['scaler_timbre_sq'])
    classifier_timbre_sq = joblib.load(REQUIRED_FILES['classifier_timbre_sq'])
    classifier_linear_timbre_sq = joblib.load(REQUIRED_FILES['classifier_linear_timbre_sq'])
    knn_timbre_sq = joblib.load(REQUIRED_FILES['knn_timbre_sq'])
    knn11_timbre_sq = joblib.load(REQUIRED_FILES['knn11_timbre_sq'])
    kmeans_fixed_timbre_sq = joblib.load(REQUIRED_FILES['kmeans_fixed_timbre_sq'])
    kmeans_timbre_sq = joblib.load(REQUIRED_FILES['kmeans_timbre_sq'])
    kmeans2_timbre_sq = joblib.load(REQUIRED_FILES['kmeans2_timbre_sq'])
    dbscan_timbre_sq = joblib.load(REQUIRED_FILES['dbscan_timbre_sq'])
    genre_kmeans_timbre_sq = joblib.load(REQUIRED_FILES['genre_kmeans_timbre_sq'])
    genre_dbscan_timbre_sq = joblib.load(REQUIRED_FILES['genre_dbscan_timbre_sq'])
    svm_on_dbscan_timbre_sq = joblib.load(REQUIRED_FILES['svm_on_dbscan_timbre_sq'])

    scaler_mid_sq = joblib.load(REQUIRED_FILES['scaler_mid_sq'])
    classifier_mid_sq = joblib.load(REQUIRED_FILES['classifier_mid_sq'])
    classifier_linear_mid_sq = joblib.load(REQUIRED_FILES['classifier_linear_mid_sq'])
    knn_mid_sq = joblib.load(REQUIRED_FILES['knn_mid_sq'])
    knn11_mid_sq = joblib.load(REQUIRED_FILES['knn11_mid_sq'])
    kmeans_fixed_mid_sq = joblib.load(REQUIRED_FILES['kmeans_fixed_mid_sq'])
    kmeans_mid_sq = joblib.load(REQUIRED_FILES['kmeans_mid_sq'])
    kmeans2_mid_sq = joblib.load(REQUIRED_FILES['kmeans2_mid_sq'])
    dbscan_mid_sq = joblib.load(REQUIRED_FILES['dbscan_mid_sq'])
    genre_kmeans_mid_sq = joblib.load(REQUIRED_FILES['genre_kmeans_mid_sq'])
    genre_dbscan_mid_sq = joblib.load(REQUIRED_FILES['genre_dbscan_mid_sq'])
    svm_on_dbscan_mid_sq = joblib.load(REQUIRED_FILES['svm_on_dbscan_mid_sq'])

    scaler_short_timbre = joblib.load(REQUIRED_FILES['scaler_short_timbre'])
    classifier_short_timbre = joblib.load(REQUIRED_FILES['classifier_short_timbre'])
    classifier_linear_short_timbre = joblib.load(REQUIRED_FILES['classifier_linear_short_timbre'])
    knn_short_timbre = joblib.load(REQUIRED_FILES['knn_short_timbre'])
    knn11_short_timbre = joblib.load(REQUIRED_FILES['knn11_short_timbre'])
    kmeans_fixed_short_timbre = joblib.load(REQUIRED_FILES['kmeans_fixed_short_timbre'])
    kmeans_short_timbre = joblib.load(REQUIRED_FILES['kmeans_short_timbre'])
    kmeans2_short_timbre = joblib.load(REQUIRED_FILES['kmeans2_short_timbre'])
    dbscan_short_timbre = joblib.load(REQUIRED_FILES['dbscan_short_timbre'])
    genre_kmeans_short_timbre = joblib.load(REQUIRED_FILES['genre_kmeans_short_timbre'])
    genre_dbscan_short_timbre = joblib.load(REQUIRED_FILES['genre_dbscan_short_timbre'])
    svm_on_dbscan_short_timbre = joblib.load(REQUIRED_FILES['svm_on_dbscan_short_timbre'])

    scaler_short_mid = joblib.load(REQUIRED_FILES['scaler_short_mid'])
    classifier_short_mid = joblib.load(REQUIRED_FILES['classifier_short_mid'])
    classifier_linear_short_mid = joblib.load(REQUIRED_FILES['classifier_linear_short_mid'])
    knn_short_mid = joblib.load(REQUIRED_FILES['knn_short_mid'])
    knn11_short_mid = joblib.load(REQUIRED_FILES['knn11_short_mid'])
    kmeans_fixed_short_mid = joblib.load(REQUIRED_FILES['kmeans_fixed_short_mid'])
    kmeans_short_mid = joblib.load(REQUIRED_FILES['kmeans_short_mid'])
    kmeans2_short_mid = joblib.load(REQUIRED_FILES['kmeans2_short_mid'])
    dbscan_short_mid = joblib.load(REQUIRED_FILES['dbscan_short_mid'])
    genre_kmeans_short_mid = joblib.load(REQUIRED_FILES['genre_kmeans_short_mid'])
    genre_dbscan_short_mid = joblib.load(REQUIRED_FILES['genre_dbscan_short_mid'])
    svm_on_dbscan_short_mid = joblib.load(REQUIRED_FILES['svm_on_dbscan_short_mid'])
    
    print("Loading complete")

    return {"song_data": song_data,
            "test_song_data": test_song_data,

            "scaler_timbre": scaler_timbre,
            "classifier_timbre": classifier_timbre,
            "classifier_linear_timbre": classifier_linear_timbre,
            "knn_timbre": knn_timbre,
            "knn11_timbre": knn11_timbre,
            "kmeans_fixed_timbre": kmeans_fixed_timbre,
            "kmeans_timbre": kmeans_timbre,
            "kmeans2_timbre": kmeans2_timbre,
            "dbscan_timbre": dbscan_timbre,
            "genre_kmeans_timbre": genre_kmeans_timbre,
            "genre_dbscan_timbre": genre_dbscan_timbre,
            "svm_on_dbscan_timbre": svm_on_dbscan_timbre,

            "scaler_mid": scaler_mid,
            "classifier_mid": classifier_mid,
            "classifier_linear_mid": classifier_linear_mid,
            "knn_mid": knn_mid,
            "knn11_mid": knn11_mid,
            "kmeans_fixed_mid": kmeans_fixed_mid,
            "kmeans_mid": kmeans_mid,
            "kmeans2_mid": kmeans2_mid,
            "dbscan_mid": dbscan_mid,
            "genre_kmeans_mid": genre_kmeans_mid,
            "genre_dbscan_mid": genre_dbscan_mid,
            "svm_on_dbscan_mid": svm_on_dbscan_mid,

            "scaler_timbre_sq": scaler_timbre_sq,
            "classifier_timbre_sq": classifier_timbre_sq,
            "classifier_linear_timbre_sq": classifier_linear_timbre_sq,
            "knn_timbre_sq": knn_timbre_sq,
            "knn11_timbre_sq": knn11_timbre_sq,
            "kmeans_fixed_timbre_sq": kmeans_fixed_timbre_sq,
            "kmeans_timbre_sq": kmeans_timbre_sq,
            "kmeans2_timbre_sq": kmeans2_timbre_sq,
            "dbscan_timbre_sq": dbscan_timbre_sq,
            "genre_kmeans_timbre_sq": genre_kmeans_timbre_sq,
            "genre_dbscan_timbre_sq": genre_dbscan_timbre_sq,
            "svm_on_dbscan_timbre_sq": svm_on_dbscan_timbre_sq,

            "scaler_mid_sq": scaler_mid_sq,
            "classifier_mid_sq": classifier_mid_sq,
            "classifier_linear_mid_sq": classifier_linear_mid_sq,
            "knn_mid_sq": knn_mid_sq,
            "knn11_mid_sq": knn11_mid_sq,
            "kmeans_fixed_mid_sq": kmeans_fixed_mid_sq,
            "kmeans_mid_sq": kmeans_mid_sq,
            "kmeans2_mid_sq": kmeans2_mid_sq,
            "dbscan_mid_sq": dbscan_mid_sq,
            "genre_kmeans_mid_sq": genre_kmeans_mid_sq,
            "genre_dbscan_mid_sq": genre_dbscan_mid_sq,
            "svm_on_dbscan_mid_sq": svm_on_dbscan_mid_sq,

            "scaler_short_timbre": scaler_short_timbre,
            "classifier_short_timbre": classifier_short_timbre,
            "classifier_linear_short_timbre": classifier_linear_short_timbre,
            "knn_short_timbre": knn_short_timbre,
            "knn11_short_timbre": knn11_short_timbre,
            "kmeans_fixed_short_timbre": kmeans_fixed_short_timbre,
            "kmeans_short_timbre": kmeans_short_timbre,
            "kmeans2_short_timbre": kmeans2_short_timbre,
            "dbscan_short_timbre": dbscan_short_timbre,
            "genre_kmeans_short_timbre": genre_kmeans_short_timbre,
            "genre_dbscan_short_timbre": genre_dbscan_short_timbre,
            "svm_on_dbscan_short_timbre": svm_on_dbscan_short_timbre,

            "scaler_short_mid": scaler_short_mid,
            "classifier_short_mid": classifier_short_mid,
            "classifier_linear_short_mid": classifier_linear_short_mid,
            "knn_short_mid": knn_short_mid,
            "knn11_short_mid": knn11_short_mid,
            "kmeans_fixed_short_mid": kmeans_fixed_short_mid,
            "kmeans_short_mid": kmeans_short_mid,
            "kmeans2_short_mid": kmeans2_short_mid,
            "dbscan_short_mid": dbscan_short_mid,
            "genre_kmeans_short_mid": genre_kmeans_short_mid,
            "genre_dbscan_short_mid": genre_dbscan_short_mid,
            "svm_on_dbscan_short_mid": svm_on_dbscan_short_mid}
