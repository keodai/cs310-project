# FMA Dataset Preprocessing - filter songs to restrict genre classes (exact match, equivalent or subgenre retained)

import logging
import sys

import paths
import os
from tinytag import TinyTag, TinyTagException
from shutil import copy2


mode = 'main_genres'  # 'empty'


# Return a utf-8 encoded string of the input
def format_string(s):
    return '' if s is None else s.encode('utf-8')


# The condition for inclusion in the restricted dataset
def include_condition(mode, genre):
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
        # Returns true if the song genre matches any of the valid genres -> song retained
        return any([True for vg in valid_genres if vg in genre.lower()])


# Recursively convert all non-hidden mp3 and wav files in source directory
# Only songs with a genre are copied to the output directory
# Run prior to setup to ensure only these songs with appropriate genre information are used
def convert_and_get_data(src_path, dst_path):
    current_dir = [(os.path.join(src_path, element), element) for element in os.listdir(src_path) if not element.startswith('.')]
    for path, element in current_dir:
        if os.path.isdir(path):
            # Recursively perform song conversion for all songs in directory
            convert_and_get_data(path, dst_path)
        elif os.path.isfile(path) and path.endswith(('.mp3', '.wav')):
            perform_copy = True
            dst = dst_path
            try:
                # Retrieve genre from song metadata using tinytag
                genre = format_string(TinyTag.get(path).genre).replace('\x00', '')
                logging.info(genre)
            except TinyTagException:
                # Do not retain the song if an exception is raised retrieving the genre
                perform_copy = False
            else:
                if not include_condition(mode, genre):
                    # Do not retain the song if its genre is not one contained in one of the defined classes
                    perform_copy = False
            finally:
                # If song is to be retained, then copy it to the new directory, if not already there. Avoids editing raw dataset.
                if perform_copy:
                    if not os.path.exists(os.path.join(dst, element)):
                        copy2(path, dst)


# Perform operation starting from source directory and setup logging
def scan(path):
    logging.basicConfig(filename="logs/filter.log", level=logging.DEBUG, format="%(asctime)s %(message)s")
    convert_and_get_data(path, paths.processed_data_path)


# Use the raw_data_path if no argument specified for the directory to use
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        scan(paths.raw_data_path)
    else:
        scan(sys.argv[1:])