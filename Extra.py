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


# sound_file = 'C:\\Users\\a\\Desktop\\Dataset\\Actor_01\\03-01-02-01-01-01-01.wav'
# sound_features(sound_file)

emotion_list = []
intensity_list = []
statement_list = []
repetition_list = []
actor_list = []
gender_list = []
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
        emotion_intensity = file_properties[3]
        statement = file_properties[4]
        repetition = file_properties[5]
        actor_number = file_properties[6]
        file_path = actor_path + '/' + audio_file
        sound_feature = sound_features(file_path)
        emotion_list.append(emotion)
        intensity_list.append(emotion_intensity)
        statement_list.append(statement)
        repetition_list.append(repetition)
        actor_list.append(actor_number)
        gender_list.append(int(actor_number) % 2)
        file_path_list.append(file_path)
        sound_features_list.append(sound_feature)

emotion_dict = {'01': 'neutral',
                '02': 'calm',
                '03': 'happy',
                '04': 'sad',
                '05': 'angry',
                '06': 'fearful',
                '07': 'disgust',
                '08': 'surprised'
                }

intensity_dict = {'01': 'normal',
                  '02': 'strong'}

statement_repetition_dict = {'01': '1st',
                             '02': '2nd'}

gender_dict = {0: 'female',
               1: 'male'
               }

emotion_dataframe = pd.DataFrame(emotion_list)
emotion_dataframe = emotion_dataframe.replace(emotion_dict)
gender_dataframe = pd.DataFrame(gender_list)
gender_dataframe = gender_dataframe.replace(gender_dict)
intensity_dataframe = pd.DataFrame(intensity_list)
intensity_dataframe = intensity_dataframe.replace(intensity_dict)
statement_dataframe = pd.DataFrame(statement_list)
statement_dataframe = statement_dataframe.replace(statement_repetition_dict)
repetition_dataframe = pd.DataFrame(repetition_list)
repetition_dataframe = repetition_dataframe.replace(statement_repetition_dict)
actor_dataframe = pd.DataFrame(actor_list)
path_dataframe = pd.DataFrame(file_path_list)
features_dataframe = pd.DataFrame(sound_features_list)
main_dataframe = pd.concat([actor_dataframe, emotion_dataframe, intensity_dataframe, statement_dataframe,
                            repetition_dataframe, gender_dataframe, path_dataframe, features_dataframe], axis=1)
main_dataframe.columns = ['actor', 'emotion', 'intensity', 'statement', 'repetition', 'gender', 'path', 'features']

main_dataframe.emotion.value_counts().plot(kind='bar')
main_dataframe.gender.value_counts().plot(kind='bar')
plt.show()
main_dataframe.to_csv('/Users/a/Desktop/audio.csv')


# b = [1, 5, 6, 2, 6, 4]
# e = [3, 7, 0, 2, 5, 6]
# a = np.array([])
# for i in b:
#     a = np.append(a, i)
#     print(a)
# c = a.argsort()[:3]
# print(c)
# d = np.array([])
# for i in c:
#     d = np.append(d, e[i])
# print(Counter(d).most_common(1)[0][0])
# print(mode(d))


# def printFeatures(dispf):
#     featuresinChromagram = dispf.loc[:,:11]
#     minOfChroma = featuresinChromagram.min().min()
#     maxOfChroma = featuresinChromagram.max().max()
#     meanOfChroma = featuresinChromagram.stack().mean()
#     chromaStdev = featuresinChromagram.stack().std()
#     print(f'Features of 12 Chromagrams: \
#     Min = {minOfChroma:.3f}, \
#     Max = {maxOfChroma:.3f}, \
#     Mean = {meanOfChroma:.3f}, \
#     Standard Deviation = {chromaStdev:.3f}')
#
#     featuresinMelspectrogram = dispf.loc[:,12:139]
#     minOfMel = featuresinMelspectrogram.min().min()
#     maxOfMel = featuresinMelspectrogram.max().max()
#     meanOfMel = featuresinMelspectrogram.stack().mean()
#     stdvOfMel = featuresinMelspectrogram.stack().std()
#     print(f'\nFeatures of 128 Melspectrograms: \
#     min = {minOfMel:.3f}, \
#     max = {maxOfMel:.3f}, \
#     mean = {meanOfMel:.3f}, \
#     Standard deviation = {stdvOfMel:.3f}')
#
#     featuresinMFCC = dispf.loc[:,140:179]
#     minOfMFCC = featuresinMFCC.min().min()
#     maxOfMFCC = featuresinMFCC.max().max()
#     meanOfMFCC = featuresinMFCC.stack().mean()
#     stdvOfMFCC = featuresinMFCC.stack().std()
#     print(f'\n40 MFCC features: \
#     min = {minOfMFCC:.2f},\
#     max = {maxOfMFCC:.2f},\
#     mean = {meanOfMFCC:.2f},\
#     deviation = {stdvOfMFCC:.3f}')
