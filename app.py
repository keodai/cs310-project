import os
import subprocess


def mp3ToWav():
    src_path = "/Users/matthew/Documents/university/CS310-Third-Year-Project/cs310-Project/samples/src/"
    dest_path = "/Users/matthew/Documents/university/CS310-Third-Year-Project/cs310-Project/samples/dest/"
    src_ext = ".mp3"
    dest_ext = ".wav"

    for file in os.listdir(src_path):
        print file
        name = file[:file.rfind(".")]
        print name
        src = src_path + name + src_ext
        dest = dest_path + name + dest_ext
        subprocess.call(["ffmpeg", "-i", src, dest])

def main():
    mp3ToWav()

if __name__ == '__main__':
    main()
