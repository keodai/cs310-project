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

