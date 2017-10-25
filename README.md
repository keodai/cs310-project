# cs310-project
## Applying Machine Learning and Audio Analysis Techniques to Music Recommendation 
### Matthew Penny [1509819]

This is the project repository, containing both the project code and documentation.

### 15/10/2017
* Started implementation, setup of GitHub repository and PyCharm IDE.
* ffmpeg required for file conversion using Pydub. Otherwise sox will need to be used in the case of stereo audio.

### 16/10/2017
* Completed method to convert files from mp3 to wav. Modified to work for all files in a directory. Could update to use pyDub.
* Resolved import issues with Anaconda environment, allowing librosa package to be installed for feature extraction.
* Performed MFCC feature extraction using the librosa package and created plots using matplotlib.pyplot.

### 17/10/2017
* Updated MFCC feature extraction and plots to be performed for each file in the directory.
* Plots are now saved to a file, rather than displayed.
* Added flag to toggle plotting behaviour on or off, so only conversion and feature extraction performed on dataset.

### 18/10/2017
* Researched STFT features. Used librosa to extract ZCR, SC, SR, SB

### 25/10/2017
* Extracted genre, title, artist and album details from MP3 metadata. 
* Genre required as it will be used as label in SVM.
* Refactored plotting.
* Feature vector now 1d list containing mean and variances of individual features.
* Scaled feature values to be in range 0-1.


### TO DO
* SoX for conversion?
* Retain metadata with wav files to be able to output the song names/artist details for recommendations/ labels.
* Raw feature extraction from frames python wave and numpy.fft.
* Yaafe, Aubio or Marsyas/WEKA for feature extraction/vector creation.
* Using pyAudioAnalysis library for feature extraction - installed dependencies numpy matplotlib scipy sklearn hmmlearn simplejson eyed3 pydub
