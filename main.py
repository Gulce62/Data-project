import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import librosa.display
import matplotlib.pyplot as plt
import os

import warnings

warnings.filterwarnings('ignore')


# Modality (01 = full-AV, 02 = video-only, 03 = audio-only) --> Always 03
# Vocal channel (01 = speech, 02 = song) --> Always 01
# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
# Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
# Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
# Repetition (01 = 1st repetition, 02 = 2nd repetition).
# Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).


def sound_features(sound_path):
    data, sampling_rate = sf.read(sound_path, dtype='float32')
    chroma = chroma_feature(data, sampling_rate)
    mfcc = mfcc_feature(data, sampling_rate)
    mel = mel_feature(data, sampling_rate)
    sound_feature_matrix = np.concatenate((chroma, mel, mfcc), axis=0, out=None)
    return sound_feature_matrix


def chroma_feature(data, sampling_rate):
    stft_data = librosa.stft(data)
    stft_data_abs = np.abs(stft_data)
    chroma_data = librosa.feature.chroma_stft(S=stft_data_abs, sr=sampling_rate)
    chroma_mean = np.mean(chroma_data.T, axis=0)
    return chroma_mean


def mfcc_feature(data, sampling_rate):
    mfcc_data = librosa.feature.mfcc(data, sampling_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc_data.T, axis=0)
    return mfcc_mean


def mel_feature(data, sampling_rate):
    mel_data = librosa.feature.melspectrogram(data, sampling_rate, n_mels=128, fmax=8000)
    mel_mean = np.mean(mel_data.T, axis=0)
    return mel_mean


emotion_list = []
actor_list = []
file_path_list = []
sound_features_list = []
audio_main_path = "/Users/a/Desktop/Dataset/"
actor_folders = os.listdir(audio_main_path)
for actor in actor_folders:
    actor_path = audio_main_path + actor
    audio_files = os.listdir(actor_path)
    for audio_file in audio_files:
        file_name = audio_file.strip().split('.')
        file_properties = file_name[0].strip().split('-')
        emotion = file_properties[2]
        file_path = actor_path + '/' + audio_file
        sound_feature = sound_features(file_path)
        print(audio_file)
        emotion_list.append(emotion)
        actor_list.append(actor)
        file_path_list.append(file_path)
        sound_features_list.append(sound_feature)
    print(actor)
emotion_dict = {'01': 'neutral',
                '02': 'calm',
                '03': 'happy',
                '04': 'sad',
                '05': 'angry',
                '06': 'fearful',
                '07': 'disgust',
                '08': 'surprised'
                }

emotion_dataframe = pd.DataFrame(emotion_list)
emotion_dataframe = emotion_dataframe.replace(emotion_dict)
actor_dataframe = pd.DataFrame(actor_list)
path_dataframe = pd.DataFrame(file_path_list)
main_dataframe = pd.concat([actor_dataframe, emotion_dataframe, path_dataframe], axis=1)
main_dataframe.columns = ['actor', 'emotion', 'path']
features_dataframe = pd.DataFrame(sound_features_list)

main_dataframe.to_csv('/Users/a/Desktop/audio.csv')
features_dataframe.to_csv('/Users/a/Desktop/features.csv')

