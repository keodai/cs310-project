import setup

import sys
import os
from sklearn.externals import joblib


def main(args):
    if (not os.path.isfile("data/classifier.pkl")
            or not os.path.isfile("data/song_data.pkl")
            or not os.path.isfile("data/scaler.pkl")):
        setup.main()

    clf = joblib.load("data/classifier.pkl")
    song_data = joblib.load("data/song_data.pkl")
    scaler = joblib.load("data/scaler.pkl")


if __name__ == '__main__':
    main(sys.argv[1:])
