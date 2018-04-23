# Perform sound recording from microphone and save to a WAV file

import pyaudio
import wave
import paths

# Setup values
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 10


# Perform recording using the defined setup values
def record(dir):
    wave_output_filename = dir + "/recording.wav"
    audio = pyaudio.PyAudio()

    # Start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Recording finished")

    full_length = frames*3  # Pad output to match length of songs in dataset

    # Stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Write the captured frames to an output wave file
    waveFile = wave.open(wave_output_filename, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(full_length))
    waveFile.close()
    return wave_output_filename


# Record to the defined upload folder
if __name__ == '__main__':
    record(paths.upload_folder)
