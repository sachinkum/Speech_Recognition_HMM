import os
from hmmlearn import hmm
from python_speech_features import mfcc
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import numpy as np
import sys



class HMM_Trainer(object):

    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    def get_score(self, input_data):
        return self.model.score(input_data)







input_folder = '/media/sachin/Data/Workspace/Python/Speech_Recognition/audio'
hmm_models = []


for dirname in os.listdir(input_folder):
    subfolder = os.path.join(input_folder, dirname)
    if not os.path.isdir(subfolder):
        continue

    label = subfolder[subfolder.rfind('/') + 1:]
    X = np.array([])
    y_words = []
    for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
        filepath = os.path.join(subfolder, filename)
        sampling_freq, audio = wavfile.read(filepath)
        mfcc_features = mfcc(audio, sampling_freq)
        if len(X) == 0:
            X = mfcc_features
        else:
            X = np.append(X, mfcc_features, axis=0)
        y_words.append(label)

    # Train and save HMM model
    hmm_trainer = HMM_Trainer()
    hmm_trainer.train(X)
    hmm_models.append((hmm_trainer, label))
    hmm_trainer = None

    # Test files
    input_files = [
        '/media/sachin/Data/Workspace/Python/Speech_Recognition/audio/pineapple/pineapple15.wav',
        '/media/sachin/Data/Workspace/Python/Speech_Recognition/audio/orange/orange15.wav',
        '/media/sachin/Data/Workspace/Python/Speech_Recognition/audio/apple/apple15.wav',
        '/media/sachin/Data/Workspace/Python/Speech_Recognition/audio/kiwi/kiwi15.wav'
    ]

    #Classify input data
    for input_file in input_files:
        # Read input file
        sampling_freq, audio = wavfile.read(input_file)
        # Extract MFCC features
        mfcc_features = mfcc(audio, sampling_freq)
        # Define variables
        max_score = -sys.maxsize
        output_label = None
        # Iterate through all HMM models and pick
        # the one with the highest score
        for item in hmm_models:
            hmm_model, label = item
            score = hmm_model.get_score(mfcc_features)
            if score > max_score:
                max_score = score
                output_label = label

        # Print the output
        print("\nTrue:", input_file[input_file.find('/') + 1:input_file.rfind('/')])
        print("Predicted:", output_label)
