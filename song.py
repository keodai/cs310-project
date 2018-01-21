from tinytag import TinyTag
import utils
import librosa
import numpy as np
import logging


def detect_pitch(y, sr):
    pitch = []
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    for t in range(len(pitches)):
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
    mean_of_mean = np.mean(frame_means, axis=axis)
    mean_of_var = np.mean(frame_vars, axis=axis)
    var_of_mean = np.var(frame_means, axis=axis)
    var_of_var = np.var(frame_vars, axis=axis)
    means = frame_means + mean_of_mean + mean_of_var
    v = frame_vars + var_of_mean + var_of_var
    return means, v


def flatten(l):
    return [item for sublist in l for item in sublist]


class Song:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.title = self.title_from_metadata()
        self.artist = self.artist_from_metadata()
        self.album = self.album_from_metadata()
        self.listed_genre = self.genre_from_metadata()
        self.predicted_genre = None
        self.features, self.timbre, self.features_sq, self.timbre_sq = self.extract_features()
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
        y, sr = librosa.load(self.dst)
        [zcr] = librosa.feature.zero_crossing_rate(y)
        [sc] = librosa.feature.spectral_centroid(y=y, sr=sr)
        [sro] = librosa.feature.spectral_rolloff(y=y, sr=sr)
        [sb] = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        pitch = detect_pitch(y, sr)

        # MuVar
        mfcc_mean = np.mean(mfcc, axis=0)
        mfcc_var = np.var(mfcc, axis=0)
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
        mfcc_m, mfcc_v = muvar_sq(mfcc, 0)
        mfcc_means = flatten(mfcc_m)
        mfcc_vars = flatten(mfcc_v)
        zcr_means, zcr_vars = muvar_sq(zcr)
        sc_means, sc_vars = muvar_sq(sc)
        sro_means, sro_vars = muvar_sq(sro)
        sb_means, sb_vars = muvar_sq(sb)
        tempo_means, tempo_vars = muvar_sq(tempo)
        pitch_means, pitch_vars = muvar_sq(pitch)

        timbre_sq = mfcc_means + mfcc_vars + \
                    zcr_means.tolist() + zcr_vars.tolist() + \
                    sc_means.tolist() + sc_vars.tolist() + \
                    sro_means.tolist() + sro_vars.tolist() + \
                    sb_means.tolist() + sb_vars.tolist()

        feature_vector_sq = mfcc_means + mfcc_vars + \
                            zcr_means.tolist() + zcr_vars.tolist() + \
                            sc_means.tolist() + sc_vars.tolist() + \
                            sro_means.tolist() + sro_vars.tolist() + \
                            sb_means.tolist() + sb_vars.tolist() + \
                            tempo_means.tolist() + tempo_vars.tolist() + \
                            pitch_means.tolist() + pitch_vars.tolist()

        return feature_vector, timbre, feature_vector_sq, timbre_sq
