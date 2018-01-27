import multi_logging
from tinytag import TinyTag
import utils
import librosa
import numpy as np
from timeit import default_timer as timer

timing = multi_logging.setup_logger('timing', 'logs/feature_times.log')
logging = multi_logging.setup_logger('output', 'logs/output.log')


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

        start = timer()
        [zcr] = librosa.feature.zero_crossing_rate(y)
        end = timer()
        zcr_time = end - start
        timing.info('ZCR: ' + str(zcr_time))

        start = timer()
        [sc] = librosa.feature.spectral_centroid(y=y, sr=sr)
        end = timer()
        sc_time = end - start
        timing.info('SC: ' + str(sc_time))

        start = timer()
        [sro] = librosa.feature.spectral_rolloff(y=y, sr=sr)
        end = timer()
        sro_time = end - start
        timing.info('SRO: ' + str(sro_time))

        start = timer()
        [sb] = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        end = timer()
        sb_time = end - start
        timing.info('SB: ' + str(sb_time))

        start = timer()
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)
        end = timer()
        mfcc_time = end - start
        timing.info('MFCC: ' + str(mfcc_time))

        start = timer()
        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        end = timer()
        tempo_time = end - start
        timing.info('TEMPO: ' + str(tempo_time))

        start = timer()
        pitch = detect_pitch(y, sr)
        end = timer()
        pitch_time = end - start
        timing.info('PITCH: ' + str(pitch_time))

        # MuVar
        start = timer()
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)
        stft = [
            np.mean(zcr), np.var(zcr),
            np.mean(sc), np.var(sc),
            np.mean(sro), np.var(sro),
            np.mean(sb), np.var(sb)
        ]
        timbre = mfcc_mean.tolist() + mfcc_var.tolist() + stft
        end = timer()
        timbre_muvar_time = end - start
        timing.info('TIMBRE_MUVAR: ' + str(timbre_muvar_time))

        start = timer()
        mid = [
            np.mean(tempo), np.var(tempo),
            np.mean(pitch), np.var(pitch)
        ]
        feature_vector = timbre + mid
        end = timer()
        mid_muvar_time = end - start
        timing.info('MID_MUVAR: ' + str(mid_muvar_time))

        total_timbre_muvar_time = zcr_time + sc_time + sro_time + sb_time + mfcc_time + timbre_muvar_time
        timing.info('Total Timbre MuVar Time: ' + str(total_timbre_muvar_time))
        total_mid_muvar_time = total_timbre_muvar_time + tempo_time + pitch_time + mid_muvar_time
        timing.info('Total Mid-level MuVar Time: ' + str(total_mid_muvar_time))
        logging.info(str(feature_vector))

        # MuVar^2
        start = timer()
        mfcc_m, mfcc_v = muvar_sq(mfcc, 1)
        mfcc_means = mfcc_m.tolist()
        mfcc_vars = mfcc_v.tolist()
        end = timer()
        mfcc_muvar2_time = end - start
        timing.info('MFCC_MUVAR2: ' + str(mfcc_muvar2_time))

        start = timer()
        zcr_means = muvar_sq(zcr)[0].tolist()
        zcr_vars = muvar_sq(zcr)[1].tolist()
        end = timer()
        zcr_muvar2_time = end - start
        timing.info('ZCR_MUVAR2: ' + str(zcr_muvar2_time))

        start = timer()
        sc_means = muvar_sq(sc)[0].tolist()
        sc_vars = muvar_sq(sc)[1].tolist()
        end = timer()
        sc_muvar2_time = end - start
        timing.info('SC_MUVAR2: ' + str(sc_muvar2_time))

        start = timer()
        sro_means = muvar_sq(sro)[0].tolist()
        sro_vars = muvar_sq(sro)[1].tolist()
        end = timer()
        sro_muvar2_time = end - start
        timing.info('SRO_MUVAR2: ' + str(sro_muvar2_time))

        start = timer()
        sb_means = muvar_sq(sb)[0].tolist()
        sb_vars = muvar_sq(sb)[1].tolist()
        end = timer()
        sb_muvar2_time = end - start
        timing.info('SB_MUVAR2: ' + str(sb_muvar2_time))

        start = timer()
        tempo_means = muvar_sq(tempo)[0].tolist()
        tempo_vars = muvar_sq(tempo)[1].tolist()
        end = timer()
        tempo_muvar2_time = end - start
        timing.info('TEMPO_MUVAR2: ' + str(tempo_muvar2_time))

        start = timer()
        pitch_means = muvar_sq(pitch)[0].tolist()
        pitch_vars = muvar_sq(pitch)[1].tolist()
        end = timer()
        pitch_muvar2_time = end - start
        timing.info('PITCH_MUVAR2: ' + str(pitch_muvar2_time))

        start = timer()
        timbre_sq = mfcc_means + mfcc_vars + zcr_means + zcr_vars + sc_means + sc_vars + sro_means + sro_vars + sb_means + sb_vars
        end = timer()
        timbre_muvar2_time = end - start
        timing.info('TIMBRE_MUVAR2: ' + str(timbre_muvar2_time))

        start = timer()
        feature_vector_sq = mfcc_means + mfcc_vars + zcr_means + zcr_vars + sc_means + sc_vars + sro_means + sro_vars + sb_means + sb_vars + tempo_means + tempo_vars + pitch_means + pitch_vars
        end = timer()
        mid_muvar2_time = end - start
        timing.info('MID_MUVAR2: ' + str(mid_muvar2_time))

        total_timbre_muvar2_time = zcr_muvar2_time + sc_muvar2_time + sro_muvar2_time + sb_muvar2_time + mfcc_muvar2_time + timbre_muvar2_time
        timing.info('Total Timbre MuVar2 Time: ' + str(total_timbre_muvar2_time))
        total_mid_muvar2_time = total_timbre_muvar2_time + tempo_muvar2_time + pitch_muvar2_time + mid_muvar2_time
        timing.info('Total Mid-level MuVar2 Time: ' + str(total_mid_muvar2_time))

        return timbre, feature_vector, timbre_sq, feature_vector_sq
