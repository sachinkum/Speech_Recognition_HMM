import os
from hmmlearn.hmm import GMMHMM
from python_speech_features import mfcc
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import numpy as np
import sys

input_folder = '/home/sachin/Downloads/cmu_us_awb_arctic-0.95-release/cmu_us_awb_arctic/wav'
hmm_models = []

X = np.array([])
for filename in os.listdir(input_folder):
    filepath = os.path.join(input_folder, filename)
    sampling_freq, audio = wavfile.read(filepath)
    mfcc_features = mfcc(audio, sampling_freq)
    if len(X) == 0:
        X = mfcc_features
    else:
        X = np.append(X, mfcc_features, axis=0)


model = GMMHMM(n_components=3, n_mix=45, n_iter=100)
X_train, X_test = train_test_split(X, train_size=0.7)
hmm_models.append(model.fit(X_train))

print(model.score(X_test))
