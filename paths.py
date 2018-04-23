# Paths for use throughout system to ensure consistency
# *** UPDATE TO CHANGE DIRECTORIES USED FOR AUDIO DATA OR FOR USE ON ANOTHER SYSTEM ***

project_audio_dir = "/Users/matthew/PycharmProjects/cs310-project/audio_data/"

# SMALL (100) DATASET (FMA)
# mode = "fma"
# training_src_path = project_audio_dir + "small_data/src/training/"
# training_dst_path = project_audio_dir + "small_data/dst/training/"
# test_src_path = project_audio_dir + "small_data/src/test/"
# test_dst_path = project_audio_dir + "small_data/dst/test/"

# MEDIUM (1000) DATASET (FMA)
# mode = "fma"
# training_src_path = project_audio_dir + "med_data/src/training/"
# training_dst_path = project_audio_dir + "med_data/dst/training/"
# test_src_path = project_audio_dir + "med_data/src/test/"
# test_dst_path = project_audio_dir + "med_data/dst/test/"


# MEDIUM (10000) DATASET (FMA)
# mode = "fma"
# training_src_path = project_audio_dir + "src/training/"
# training_dst_path = project_audio_dir + "dst/training/"
# test_src_path = project_audio_dir + "src/test/"
# test_dst_path = project_audio_dir + "dst/test/"

# FULL DATASET (FMA)
# mode = "fma"
# training_src_path = project_audio_dir + "src_full/training/"
# training_dst_path = project_audio_dir + "dst_full/training/"
# test_src_path = project_audio_dir + "src_full/test/"
# test_dst_path = project_audio_dir + "dst_full/test/"

# MEDIUM (10000) DATASET (FMA RESTRICTED GENRES)
mode = "fma"
training_src_path = project_audio_dir + "restricted/src/training/"
training_dst_path = project_audio_dir + "restricted/dst/training/"
test_src_path = project_audio_dir + "restricted/src/test/"
test_dst_path = project_audio_dir + "restricted/dst/test/"

# DATASET 2 (GTZAN)
# mode = "ds2"
# training_src_path = project_audio_dir + "ds2/"
# training_dst_path = project_audio_dir + "ds2/training/"
# test_src_path = project_audio_dir + "ds2t/"
# test_dst_path = project_audio_dir + "ds2t/test/"

# PERMANENT PATHS
# FMA Info
raw_data_path = project_audio_dir + "raw_dataset/"
processed_data_path = project_audio_dir + "processed/"
data_path_exclude = project_audio_dir + "no_genre/"

# GTZAN info
ds2_src = project_audio_dir + "gtzan/"
ds2_dst = project_audio_dir + "ds2/"

# App info
dst_ext = ".wav"
output_dir = project_audio_dir + "converted/"

upload_folder = project_audio_dir + "data/uploads/"
plot_dir = project_audio_dir + "plots/"
