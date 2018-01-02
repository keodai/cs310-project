from tinytag import TinyTag
import utils
import librosa
import numpy as np
import logging


class Song:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.title = self.title_from_metadata()
        self.artist = self.artist_from_metadata()
        self.album = self.album_from_metadata()
        self.listed_genre = self.genre_from_metadata()
        self.predicted_genre = None
        self.features = self.extract_features()
        self.normalised_features = None
        self.dbscan_cluster_id = None

    def genre_from_metadata(self):
        return utils.format_string(TinyTag.get(self.src).genre).replace('\x00', '')

    def title_from_metadata(self):
        return utils.format_string(TinyTag.get(self.src).title).replace('\x00', '')

    def artist_from_metadata(self):
        return utils.format_string(TinyTag.get(self.src).artist).replace('\x00', '')

    def album_from_metadata(self):
        return utils.format_string(TinyTag.get(self.src).album).replace('\x00', '')

    def extract_features(self):
        ysr = librosa.load(self.dst)
        zcr = librosa.feature.zero_crossing_rate(ysr[0])
        sc = librosa.feature.spectral_centroid(y=ysr[0], sr=ysr[1])
        sr = librosa.feature.spectral_rolloff(y=ysr[0], sr=ysr[1])
        sb = librosa.feature.spectral_bandwidth(y=ysr[0], sr=ysr[1])
        mfcc = librosa.feature.mfcc(y=ysr[0], sr=ysr[1])
        feature_vector = [
            np.mean(zcr), np.var(zcr),
            np.mean(sc), np.var(sc),
            np.mean(sr), np.var(sr),
            np.mean(sb), np.var(sb),
            np.mean(mfcc), np.var(mfcc)
        ]
        logging.info(str(feature_vector))
        return feature_vector
