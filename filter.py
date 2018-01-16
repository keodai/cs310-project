import logging
import sys

from sklearn.externals import joblib

import converter
import paths
import os
import utils
# import setup
from song import Song
from tinytag import TinyTag, TinyTagException

song_data = []


def single_song(src, dst_path):
    logging.info("Processing: " + src)
    dst = converter.convert(src, dst_path, paths.dst_ext)
    return Song(src, dst)


def convert_and_get_data(src_path, dst_path):
    current_dir = [(os.path.join(src_path, element), element) for element in os.listdir(src_path) if not element.startswith('.')]
    for path, element in current_dir:
        if os.path.isdir(path):
            convert_and_get_data(path, dst_path)
        elif os.path.isfile(path):
            # song_data.append(single_song(path, dst_path))
            dst = dst_path
            try:
                genre = utils.format_string(TinyTag.get(path).genre).replace('\x00', '')
                logging.info(genre)
            except TinyTagException:
                dst = paths.data_path_exclude
            else:
                if genre == "":
                    dst = paths.data_path_exclude
            finally:
                converter.convert(path, dst, paths.dst_ext)


def check_metadata(filepath):
    pass


def scan(path):
    global song_data
    logging.basicConfig(filename="logs/output.log", level=logging.DEBUG, format="%(asctime)s %(message)s")
    # for path in utils.visible(path):
    #     if os.path.isfile(path):
    #         filepath = converter.convert(path, paths.processed_data_path, paths.dst_ext)
    #         check_metadata(filepath)
    #     elif os.path.isfile(path):
    #         scan(path)
    #     else:
    #         print("Not a file or directory")
    #         exit(1)
    song_data_paths = "data/song_data.pkl"
    # if os.path.isfile(song_data_paths):
    #     song_data = joblib.load(song_data_paths)
    # else:
    convert_and_get_data(paths.raw_data_path, paths.processed_data_path)

    # song_data_genre_present, song_data_no_genre = [], []
    # for song in song_data:
    #     (song_data_genre_present if song.listed_genre != "" else song_data_no_genre).append(song)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        scan(paths.raw_data_path)
    else:
        scan(sys.argv[1:])