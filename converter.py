import os
import subprocess
import logging


def convert(src, dst_dir, dst_ext):
    base = os.path.basename(src)
    dst = dst_dir + os.path.splitext(base)[0] + dst_ext
    try:
        # proc = \
        subprocess.check_output(["ffmpeg", "-y", "-i", src, dst], stderr=subprocess.STDOUT)
        return dst
    except subprocess.CalledProcessError as e:
        logging.error(e)
        return
