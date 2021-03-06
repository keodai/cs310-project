# Perform feature extraction and create an object per song

import multi_logging
import paths
from tinytag import TinyTag 
import librosa
import numpy as np
from timeit import default_timer as timer
import os

timing = multi_logging.setup_logger('timing', 'logs/feature_times.log')

# Genre aliases for FMA_R
rock_aliases = ['rock', 'punk', 'indie', 'metal', 'alternative']
elec_aliases = ['electronic', 'disco', 'dance', 'drum and bass', 'techno', 'electro', 'garage']
folk_aliases = ['folk']
hh_aliases = ['hip-hop', 'rap']
pop_aliases = ['pop']
noise_aliases = ['noise']
classic_aliases = ['classical']
acapella_aliases = ['acapella']
jazz_aliases = ['jazz']
blues_aliases = ['blues']
country_aliases = ['country']
reggae_aliases = ['reggae']
instr_aliases = ['instrumental', 'band', 'acoustic', 'percussion']


# Return the input as a utf-8 encoded string
def format_string(s):
    return '' if s is None else s.encode('utf-8')


# Extract pitch features from wave
def detect_pitch(y, sr):
    pitch = []
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    for t in range(0, len(pitches[0])):
        index = magnitudes[:, t].argmax()
        pitch.append(pitches[index, t])
    return np.asarray(pitch)


# Split features into sublists for MuVar^2 feature extraction
def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


# Extract MuVar^2 summary of a given feature
def muvar_sq(feature, axis=None):
    frame_means = []
    frame_vars = []
    frames = chunk_it(feature, 10)
    # Create vectors of MFCC means and MFCC variances
    for frame in frames:
        frame_means.append(np.mean(frame, axis=axis))
        frame_vars.append(np.var(frame, axis=axis))
    frame_means = frame_means[:10]
    frame_vars = frame_means[:10]
    # Take the mean and variance of each vector
    mean_of_mean = np.mean(frame_means, axis=axis)
    mean_of_var = np.mean(frame_vars, axis=axis)
    var_of_mean = np.var(frame_means, axis=axis)
    var_of_var = np.var(frame_vars, axis=axis)
    means = np.append(np.array(frame_means), [mean_of_mean, mean_of_var])
    v = np.append(np.array(frame_vars), [var_of_mean, var_of_var])
    return means, v


# Container for all information relating to an individual song
class Song:
    def __init__(self, src, dst):
        # Attributes for location, metadata, features and basic predictions from setup time
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
        self.predicted_genre_short_features = None
        self.predicted_genre_short_timbre = None
        self.timbre, self.features, self.timbre_sq, self.features_sq, self.short_timbre, self.short_features = self.extract_features()
        self.normalised_features = None
        self.normalised_timbre = None
        self.normalised_features_sq = None
        self.normalised_timbre_sq = None
        self.normalised_short_features = None
        self.normalised_short_timbre = None
        self.dbscan_cluster_id_timbre = None
        self.dbscan_cluster_id_features = None
        self.dbscan_cluster_id_timbre_sq = None
        self.dbscan_cluster_id_features_sq = None
        self.dbscan_cluster_id_short_timbre = None
        self.dbscan_cluster_id_short_features = None

    # Retrieve the genre information from the metadata or directory depending on dataset
    def genre_from_metadata(self):
        if paths.mode == "ds2":
            # GTZAN dataset genres are given by the directory name
            fname = os.path.splitext(os.path.basename(self.src))[0]
            return fname.split(".")[0]
        else:
            # FMA genres can be extracted from the metadata
            genre = format_string(TinyTag.get(self.src).genre).replace('\x00', '').lower()

            # Check which of the aliases are present and assign the corresponding genre label
            if any([alias in genre for alias in rock_aliases]):
                genre = 'rock'
            elif any([alias in genre for alias in elec_aliases]):
                genre = 'electronic'
            elif any([alias in genre for alias in folk_aliases]):
                genre = 'folk'
            elif any([alias in genre for alias in hh_aliases]):
                genre = 'hip-hop'
            elif any([alias in genre for alias in pop_aliases]):
                genre = 'pop'
            elif any([alias in genre for alias in noise_aliases]):
                genre = 'noise'
            elif any([alias in genre for alias in classic_aliases]):
                genre = 'classical'
            elif any([alias in genre for alias in acapella_aliases]):
                genre = 'acapella'
            elif any([alias in genre for alias in jazz_aliases]):
                genre = 'jazz'
            elif any([alias in genre for alias in blues_aliases]):
                genre = 'blues'
            elif any([alias in genre for alias in country_aliases]):
                genre = 'country'
            elif any([alias in genre for alias in reggae_aliases]):
                genre = 'reggae'
            elif any([alias in genre for alias in instr_aliases]):
                genre = 'instrumental'

            return genre

    # Retrieve title information - in metadata for FMA, placeholder for GTZAN
    def title_from_metadata(self):
        if paths.mode == "ds2":
            return os.path.splitext(os.path.basename(self.src))[0]
        else:
            return format_string(TinyTag.get(self.src).title).replace('\x00', '')

    # Retrieve artist information - in metadata for FMA, placeholder for GTZAN
    def artist_from_metadata(self):
        if paths.mode == "ds2":
            return os.path.splitext(os.path.basename(self.src))[0]
        else:
            return format_string(TinyTag.get(self.src).artist).replace('\x00', '')

    # Retrieve album information - in metadata for FMA, placeholder for GTZAN
    def album_from_metadata(self):
        if paths.mode == "ds2":
            return os.path.splitext(os.path.basename(self.src))[0]
        else:
            return format_string(TinyTag.get(self.src).album).replace('\x00', '')

    # Perform timed extraction of all features
    def extract_features(self):
        y, sr = librosa.load(self.dst)

        # LOW-LEVEL TIMBRE FEATURES
        # Zero Crossing Rate (ZCR)
        start = timer()
        [zcr] = librosa.feature.zero_crossing_rate(y)
        end = timer()
        zcr_time = end - start
        timing.info('ZCR: ' + str(zcr_time))

        # Spectral Centroid (SC)
        start = timer()
        [sc] = librosa.feature.spectral_centroid(y=y, sr=sr)
        end = timer()
        sc_time = end - start
        timing.info('SC: ' + str(sc_time))

        # Spectral Rolloff (SR)
        start = timer()
        [sro] = librosa.feature.spectral_rolloff(y=y, sr=sr)
        end = timer()
        sro_time = end - start
        timing.info('SRO: ' + str(sro_time))

        # Spectral Bandwidth (SB)
        start = timer()
        [sb] = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        end = timer()
        sb_time = end - start
        timing.info('SB: ' + str(sb_time))

        # MFCC
        start = timer()
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=100)
        end = timer()
        mfcc_time = end - start
        timing.info('MFCC: ' + str(mfcc_time))

        # MID-LEVEL FEATURES
        # Tempo
        start = timer()
        onset_env = librosa.onset.onset_strength(y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
        end = timer()
        tempo_time = end - start
        timing.info('TEMPO: ' + str(tempo_time))

        # Pitch
        start = timer()
        pitch = detect_pitch(y, sr)
        end = timer()
        pitch_time = end - start
        timing.info('PITCH: ' + str(pitch_time))

        # TEMPORAL FEATURES
        # MuVar
        start = timer()
        mfcc_mean = np.sort(np.mean(mfcc, axis=1))
        mfcc_var = np.sort(np.var(mfcc, axis=1))

        # Create feature vectors by concatenating individual features
        # MuVar Timbre Feature Vector
        stft = np.array([
            np.mean(zcr), np.var(zcr),
            np.mean(sc), np.var(sc),
            np.mean(sro), np.var(sro),
            np.mean(sb), np.var(sb)
        ])
        timbre = np.concatenate((mfcc_mean, mfcc_var, stft))
        end = timer()
        timbre_muvar_time = end - start
        timing.info('TIMBRE_MUVAR: ' + str(timbre_muvar_time))

        # Short Timbre Feature Vector
        start = timer()
        short_timbre = np.concatenate((np.array([np.mean(mfcc_mean), np.var(mfcc_mean), np.mean(mfcc_var), np.var(mfcc_var)]),stft))
        end = timer()
        s_timbre_muvar_time = end - start
        timing.info('SHORT_TIMBRE_MUVAR: ' + str(s_timbre_muvar_time))

        # MuVar Mid-Level Feature Vector
        start = timer()
        mid = np.array([
            np.mean(tempo), np.var(tempo),
            np.mean(pitch), np.var(pitch)
        ])
        feature_vector = np.append(timbre, mid)
        end = timer()
        mid_muvar_time = end - start
        timing.info('MID_MUVAR: ' + str(mid_muvar_time))

        # Short Mid-Level Feature Vector
        start = timer()
        short_feature_vector = np.append(short_timbre, mid)
        end = timer()
        s_mid_muvar_time = end - start
        timing.info('SHORT_MID_MUVAR: ' + str(s_mid_muvar_time))

        # Record overall time (MuVar)
        total_timbre_muvar_time = zcr_time + sc_time + sro_time + sb_time + mfcc_time + timbre_muvar_time
        timing.info('Total Timbre MuVar Time: ' + str(total_timbre_muvar_time))
        total_mid_muvar_time = total_timbre_muvar_time + tempo_time + pitch_time + mid_muvar_time
        timing.info('Total Mid-level MuVar Time: ' + str(total_mid_muvar_time))

        # MuVar^2
        # MFCC
        start = timer()
        mfcc_means, mfcc_vars = muvar_sq(mfcc, 1)
        end = timer()
        mfcc_muvar2_time = end - start
        timing.info('MFCC_MUVAR2: ' + str(mfcc_muvar2_time))

        # ZCR
        start = timer()
        zcr_means, zcr_vars = muvar_sq(zcr)
        end = timer()
        zcr_muvar2_time = end - start
        timing.info('ZCR_MUVAR2: ' + str(zcr_muvar2_time))

        # SC
        start = timer()
        sc_means, sc_vars = muvar_sq(sc)
        end = timer()
        sc_muvar2_time = end - start
        timing.info('SC_MUVAR2: ' + str(sc_muvar2_time))

        # SR
        start = timer()
        sro_means, sro_vars = muvar_sq(sro)
        end = timer()
        sro_muvar2_time = end - start
        timing.info('SRO_MUVAR2: ' + str(sro_muvar2_time))

        # SB
        start = timer()
        sb_means, sb_vars = muvar_sq(sb)
        end = timer()
        sb_muvar2_time = end - start
        timing.info('SB_MUVAR2: ' + str(sb_muvar2_time))

        # Tempo
        start = timer()
        tempo_means, tempo_vars = muvar_sq(tempo)
        end = timer()
        tempo_muvar2_time = end - start
        timing.info('TEMPO_MUVAR2: ' + str(tempo_muvar2_time))

        # Pitch
        start = timer()
        pitch_means, pitch_vars = muvar_sq(pitch)
        end = timer()
        pitch_muvar2_time = end - start
        timing.info('PITCH_MUVAR2: ' + str(pitch_muvar2_time))

        # MuVar^2 Timbre Feature Vector
        start = timer()
        timbre_sq = np.concatenate((np.sort(mfcc_means), np.sort(mfcc_vars), np.sort(zcr_means), np.sort(zcr_vars), np.sort(sc_means), np.sort(sc_vars), np.sort(sro_means), np.sort(sro_vars), np.sort(sb_means), np.sort(sb_vars)))
        end = timer()
        timbre_muvar2_time = end - start
        timing.info('TIMBRE_MUVAR2: ' + str(timbre_muvar2_time))

        # MuVar^2 Mid-Level Feature Vector
        start = timer()
        feature_vector_sq = np.concatenate((timbre_sq, np.sort(tempo_means), np.sort(tempo_vars), np.sort(pitch_means), np.sort(pitch_vars)))
        end = timer()
        mid_muvar2_time = end - start
        timing.info('MID_MUVAR2: ' + str(mid_muvar2_time))

        # Record overall time (MuVar^2)
        total_timbre_muvar2_time = zcr_muvar2_time + sc_muvar2_time + sro_muvar2_time + sb_muvar2_time + mfcc_muvar2_time + timbre_muvar2_time
        timing.info('Total Timbre MuVar2 Time: ' + str(total_timbre_muvar2_time))
        total_mid_muvar2_time = total_timbre_muvar2_time + tempo_muvar2_time + pitch_muvar2_time + mid_muvar2_time
        timing.info('Total Mid-level MuVar2 Time: ' + str(total_mid_muvar2_time))

        return timbre, feature_vector, timbre_sq, feature_vector_sq, short_timbre, short_feature_vector

    # Setters for each attribute depending on the vector type
    def set_normalised_features(self, vector_type, value):
        if vector_type == "TIMBRE":
            self.normalised_timbre = value
        elif vector_type == "MID":
            self.normalised_features = value
        elif vector_type == "TIMBRE_SQ":
            self.normalised_timbre_sq = value
        elif vector_type == "MID_SQ":
            self.normalised_features_sq = value
        elif vector_type == "SHORT_TIMBRE":
            self.normalised_short_timbre = value
        elif vector_type == "SHORT_MID":
            self.normalised_short_features = value
        else:
            raise ValueError('Invalid vector type specified')

    def set_predicted_genre(self, vector_type, value):
        if vector_type == "TIMBRE":
            self.predicted_genre_timbre = value
        elif vector_type == "MID":
            self.predicted_genre_features = value
        elif vector_type == "TIMBRE_SQ":
            self.predicted_genre_timbre_sq = value
        elif vector_type == "MID_SQ":
            self.predicted_genre_features_sq = value
        elif vector_type == "SHORT_TIMBRE":
            self.predicted_genre_short_timbre = value
        elif vector_type == "SHORT_MID":
            self.predicted_genre_short_features = value
        else:
            raise ValueError('Invalid vector type specified')

    def set_dbscan_cluster_id(self, vector_type, value):
        if vector_type == "TIMBRE":
            self.dbscan_cluster_id_timbre = value
        elif vector_type == "MID":
            self.dbscan_cluster_id_features = value
        elif vector_type == "TIMBRE_SQ":
            self.dbscan_cluster_id_timbre_sq = value
        elif vector_type == "MID_SQ":
            self.dbscan_cluster_id_features_sq = value
        elif vector_type == "SHORT_TIMBRE":
            self.dbscan_cluster_id_short_timbre = value
        elif vector_type == "SHORT_MID":
            self.dbscan_cluster_id_short_features = value
        else:
            raise ValueError('Invalid vector type specified')

    # Getters for each attribute depending on the vector type
    def get_predicted_genre(self, vector_type):
        if vector_type == "TIMBRE":
            return self.predicted_genre_timbre
        elif vector_type == "MID":
            return self.predicted_genre_features
        elif vector_type == "TIMBRE_SQ":
            return self.predicted_genre_timbre_sq
        elif vector_type == "MID_SQ":
            return self.predicted_genre_features_sq
        elif vector_type == "SHORT_TIMBRE":
            return self.predicted_genre_short_timbre
        elif vector_type == "SHORT_MID":
            return self.predicted_genre_short_features
        else:
            raise ValueError('Invalid vector type specified')

    def get_normalised_features(self, vector_type):
        if vector_type == "TIMBRE":
            return self.normalised_timbre
        elif vector_type == "MID":
            return self.normalised_features
        elif vector_type == "TIMBRE_SQ":
            return self.normalised_timbre_sq
        elif vector_type == "MID_SQ":
            return self.normalised_features_sq
        elif vector_type == "SHORT_TIMBRE":
            return self.normalised_short_timbre
        elif vector_type == "SHORT_MID":
            return self.normalised_short_features
        else:
            raise ValueError('Invalid vector type specified')

    def get_dbscan_cluster_id(self, vector_type):
        if vector_type == "TIMBRE":
            return self.dbscan_cluster_id_timbre
        elif vector_type == "MID":
            return self.dbscan_cluster_id_features
        elif vector_type == "TIMBRE_SQ":
            return self.dbscan_cluster_id_timbre_sq
        elif vector_type == "MID_SQ":
            return self.dbscan_cluster_id_features_sq
        elif vector_type == "SHORT_TIMBRE":
            return self.dbscan_cluster_id_short_timbre
        elif vector_type == "SHORT_MID":
            return self.dbscan_cluster_id_short_features
        else:
            raise ValueError('Invalid vector type specified')

    def get_features(self, vector_type):
        if vector_type == "TIMBRE":
            return self.timbre
        elif vector_type == "MID":
            return self.features
        elif vector_type == "TIMBRE_SQ":
            return self.timbre_sq
        elif vector_type == "MID_SQ":
            return self.features_sq
        elif vector_type == "SHORT_TIMBRE":
            return self.short_timbre
        elif vector_type == "SHORT_MID":
            return self.short_features
        else:
            raise ValueError('Invalid vector type specified')
