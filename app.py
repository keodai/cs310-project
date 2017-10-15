import subprocess

src_path = "/Users/matthew/Documents/university/CS310-Third-Year-Project/cs310-Project/samples/src/"
dest_path = "/Users/matthew/Documents/university/CS310-Third-Year-Project/cs310-Project/samples/dest/"
src_ext = "mp3"
dest_ext = "wav"


def mp3ToWav(file):
    subprocess.call(['ffmpeg', '-i', src_path + file + "." + src_ext, dest_path + file + "." + dest_ext])

def main():
    mp3ToWav("000002")

if __name__ == '__main__':
    main()
