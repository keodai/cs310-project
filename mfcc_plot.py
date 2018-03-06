import librosa
import librosa.display
import matplotlib.pyplot as plt

import paths

y, sr = librosa.load(paths.training_dst_path + "000462.wav")
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.savefig(paths.plot_dir + "mfccblues.png")
