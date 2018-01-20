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


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


class Song:
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.title = self.title_from_metadata()
        self.artist = self.artist_from_metadata()
        self.album = self.album_from_metadata()
        self.listed_genre = self.genre_from_metadata()
        self.predicted_genre = None
        self.features, self.timbre_features = self.extract_features()
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
        zcr = librosa.feature.zero_crossing_rate(y)
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        sro = librosa.feature.spectral_rolloff(y=y, sr=sr)
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
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

        timbre = mfcc_mean + mfcc_var + stft
        feature_vector = mfcc_mean + mfcc_var + stft + mid
        logging.info(str(feature_vector))

        # MuVar^2
        # todo: refactor this into a method for each feature, with optional param for axis.
        mfcc_frame_means = []
        mfcc_frame_vars = []
        mfcc_frames = chunkIt(mfcc, 10)
        for frame in mfcc_frames:
            mfcc_frame_means.append(np.mean(frame, axis=0))
            mfcc_frame_vars.append(np.var(frame, axis=0))
        mfcc_mean_of_mean = np.mean(mfcc_frame_means, axis=0)
        mfcc_mean_of_var = np.mean(mfcc_frame_vars, axis=0)
        mfcc_var_of_mean = np.var(mfcc_frame_means, axis=0)
        mfcc_var_of_var = np.var(mfcc_frame_vars, axis=0)

        zcr_frames = chunkIt(zcr, 10)
        sc_frames = chunkIt(sc, 10)
        sro_frames = chunkIt(sro, 10)
        sb_frames = chunkIt(sb, 10)
        tempo_frames = chunkIt(tempo, 10)
        pitch_frames = chunkIt(pitch, 10)





        return feature_vector, timbre
