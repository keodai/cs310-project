from tinytag import TinyTag
import utils
import librosa
import numpy as np
import logging


def detect_pitch(y, sr):
    pitch = []
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    for t in range(0, len(pitches[0])):
        index = magnitudes[:, t].argmax()
        pitch.append(pitches[index, t])
    return pitch


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def muvar_sq(feature, axis=None):
    frame_means = []
    frame_vars = []
    frames = chunk_it(feature, 10)
    for frame in frames:
        frame_means.append(np.mean(frame, axis=axis))
        frame_vars.append(np.var(frame, axis=axis))
    frame_means = frame_means[:10]
    frame_vars = frame_means[:10]
    mean_of_mean = np.mean(frame_means, axis=axis)
    mean_of_var = np.mean(frame_vars, axis=axis)
    var_of_mean = np.var(frame_means, axis=axis)
    var_of_var = np.var(frame_vars, axis=axis)
    means = np.append(np.array(frame_means), [mean_of_mean, mean_of_var])
    v = np.append(np.array(frame_vars), [var_of_mean, var_of_var])
    return means, v


def flatten(l):
    return np.array([item for sublist in l for item in sublist])


class Song:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.title = self.title_from_metadata()
        self.artist = self.artist_from_metadata()
        self.album = self.album_from_metadata()
        self.listed_genre = self.genre_from_metadata()
        self.predicted_genre_features = None
        self.predicted_genre_timbre = None
        self.predicted_genre_features_sq = None
        self.predicted_genre_timbre_sq = None
        self.timbre, self.features, self.timbre_sq, self.features_sq = self.extract_features()
        self.normalised_features = None
        self.normalised_timbre = None
        self.normalised_features_sq = None
        self.normalised_timbre_sq = None
        self.dbscan_cluster_id_timbre = None
        self.dbscan_cluster_id_features = None
        self.dbscan_cluster_id_timbre_sq = None
        self.dbscan_cluster_id_timbre_features_sq = None

    def genre_from_metadata(self):
        return utils.format_string(TinyTag.get(self.src).genre).replace('\x00', '')

    def title_from_metadata(self):
        return utils.format_string(TinyTag.get(self.src).title).replace('\x00', '')

    def artist_from_metadata(self):
        return utils.format_string(TinyTag.get(self.src).artist).replace('\x00', '')

    def album_from_metadata(self):
        return utils.format_string(TinyTag.get(self.src).album).replace('\x00', '')

    def extract_features(self):
        y, sr = librosa.load(self.dst)
        [zcr] = librosa.feature.zero_crossing_rate(y)
        [sc] = librosa.feature.spectral_centroid(y=y, sr=sr)
        [sro] = librosa.feature.spectral_rolloff(y=y, sr=sr)
        [sb] = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)

        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        pitch = detect_pitch(y, sr)

        # MuVar
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)
        stft = [
            np.mean(zcr), np.var(zcr),
            np.mean(sc), np.var(sc),
            np.mean(sro), np.var(sro),
            np.mean(sb), np.var(sb)
        ]
        mid = [
            np.mean(tempo), np.var(tempo),
            np.mean(pitch), np.var(pitch)
        ]

        timbre = mfcc_mean.tolist() + mfcc_var.tolist() + stft
        feature_vector = timbre + mid
        logging.info(str(feature_vector))

        # MuVar^2
        mfcc_m, mfcc_v = muvar_sq(mfcc, 1)
        mfcc_means = mfcc_m.tolist()
        mfcc_vars = mfcc_v.tolist()
        zcr_means = muvar_sq(zcr)[0].tolist()
        zcr_vars = muvar_sq(zcr)[1].tolist()
        sc_means = muvar_sq(sc)[0].tolist()
        sc_vars = muvar_sq(sc)[1].tolist()
        sro_means = muvar_sq(sro)[0].tolist()
        sro_vars = muvar_sq(sro)[1].tolist()
        sb_means = muvar_sq(sb)[0].tolist()
        sb_vars = muvar_sq(sb)[1].tolist()
        tempo_means = muvar_sq(tempo)[0].tolist()
        tempo_vars = muvar_sq(tempo)[1].tolist()
        pitch_means = muvar_sq(pitch)[0].tolist()
        pitch_vars = muvar_sq(pitch)[1].tolist()

        timbre_sq = mfcc_means + mfcc_vars + zcr_means + zcr_vars + sc_means + sc_vars + sro_means + sro_vars + sb_means + sb_vars
        feature_vector_sq = mfcc_means + mfcc_vars + zcr_means + zcr_vars + sc_means + sc_vars + sro_means + sro_vars + sb_means + sb_vars + tempo_means + tempo_vars + pitch_means + pitch_vars

        return timbre, feature_vector, timbre_sq, feature_vector_sq
