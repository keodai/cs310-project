import logging
import sys

import paths
import os
from tinytag import TinyTag, TinyTagException
from shutil import copy2


mode = 'main_genres'  # 'empty'


# Return a utf-8 encoded string, of the input
def format_string(s):
    return '' if s is None else s.encode('utf-8')


def exclude_condition(mode, genre):
    if mode == 'empty':
        return genre == ""
    else:
        valid_genres = ['rock', 'punk', 'indie', 'metal', 'alternative',
                        'electronic', 'disco', 'dance', 'drum and bass', 'techno', 'electro', 'garage',
                        'folk',
                        'hip-hop', 'rap',
                        'pop',
                        'noise',
                        'classical',
                        'acapella',
                        'jazz',
                        'blues',
                        'country',
                        'reggae',
                        'instrumental', 'band', 'acoustic', 'percussion']
        return any([True for vg in valid_genres if vg in genre.lower()])


# Recursively convert all non-hidden mp3 and wav files in source directory
# Only songs with a genre are copied to the output directory
# Run prior to setup to ensure only these songs with genre information are used
def convert_and_get_data(src_path, dst_path):
    current_dir = [(os.path.join(src_path, element), element) for element in os.listdir(src_path) if not element.startswith('.')]
    for path, element in current_dir:
        if os.path.isdir(path):
            convert_and_get_data(path, dst_path)
        elif os.path.isfile(path) and path.endswith(('.mp3', '.wav')):
            dst = dst_path
            try:
                genre = format_string(TinyTag.get(path).genre).replace('\x00', '')
                logging.info(genre)
            except TinyTagException:
                dst = paths.data_path_exclude
            else:
                if exclude_condition(mode, genre):
                    dst = paths.data_path_exclude
            finally:
                if not os.path.exists(os.path.join(dst, element)):
                    copy2(path, dst)


def scan(path):
    logging.basicConfig(filename="logs/filter.log", level=logging.DEBUG, format="%(asctime)s %(message)s")
    convert_and_get_data(path, paths.processed_data_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        scan(paths.raw_data_path)
    else:
        scan(sys.argv[1:])