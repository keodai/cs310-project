# GTZAN Dataset Preprocessing - Changes the format of dataset 2 to match original dataset by putting genre information from directory into metadata

import os
import subprocess
import logging

import multi_logging
import paths

cv = multi_logging.setup_logger('ds2', 'logs/ds2.log')


# Return paths to non-hidden files in a directory, excluding sub-directories
def visible(src_path):
    return [os.path.join(src_path, file)
            for file in os.listdir(src_path)
            if not file.startswith('.') and os.path.isfile(os.path.join(src_path, file))]


# Retrieve the path for each genre directory
def directories(src_path):
    return [genre_dir for genre_dir in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, genre_dir))]


# Perform conversion and data/feature extraction on all songs in a directory
def convert_all(src_path, dst_path):
    for genre in directories(src_path):
        for src in visible(os.path.join(src_path, genre)):
            convert(src, dst_path, genre)


# Convert input file to mp3, prior to use in existing FMA code. Put genre name from directory into metadata.
def convert(src, dst_dir, genre):
    base = os.path.basename(src)
    name = os.path.splitext(base)[0]
    name = name.replace(".", "")
    dst = dst_dir + name + '.mp3'
    try:
        subprocess.check_output(["ffmpeg", "-y", "-i", src, "-acodec", "libmp3lame",
                                 "-metadata", "genre=" + genre, "-metadata", "title=" + name,
                                 "-metadata",  "artist=" + name, "-metadata", "album=" + name, dst],
                                stderr=subprocess.STDOUT)
        return dst
    except subprocess.CalledProcessError as e:
        logging.error(e)
        return


# Convert all songs in ds2 source to destination directory
if __name__ == "__main__":
    convert_all(paths.ds2_src, paths.ds2_dst)