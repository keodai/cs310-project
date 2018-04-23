# Additional utility for plotting MFCC spectrograms

import librosa.display
import matplotlib.pyplot as plt

import paths

# Load song
y, sr = librosa.load(paths.training_dst_path + "000721.wav")  # Examples: 000462 000567 000721 000831

# Extract MFCC Features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

# Plot figure
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time', vmin=-300, vmax=200)
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.savefig("mfcc.png")
