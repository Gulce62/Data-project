import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
import warnings
import librosa.display as ld
from matplotlib.colors import Normalize

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
    sd.play(data, sampling_rate)
    status = sd.wait()  # Wait until file is done playing
    plt.figure(figsize=(8, 4))
    librosa.display.waveshow(data, sr=sampling_rate)
    plt.title('Waveplot - Male Calm')
    plt.show()
    chroma_feature(data, sampling_rate)
    mfcc_feature(data, sampling_rate)
    mel_feature(data, sampling_rate)


def chroma_feature(data, sampling_rate):
    stft_data = librosa.stft(data)
    stft_data_abs = np.abs(stft_data)
    chroma_data = librosa.feature.chroma_stft(S=stft_data_abs, sr=sampling_rate)
    stft_data_db = librosa.amplitude_to_db(stft_data_abs, ref=np.max)
    figure, axes = plt.subplots()
    image_stft = librosa.display.specshow(stft_data_db, y_axis='log', x_axis='time', sr=sampling_rate)
    plt.title('Logarithmic STFT Spectrum Plot of Neutral Sound')
    figure.colorbar(image_stft, ax=axes, format="%+2.0f dB")
    plt.show()
    figure, axes = plt.subplots()
    image_croma = librosa.display.specshow(chroma_data, y_axis='chroma', x_axis='time')
    plt.title('Chroma Spectrum Plot of Neutral Sound')
    figure.colorbar(image_croma, ax=axes)
    plt.show()


def mfcc_feature(data, sampling_rate):
    mfcc_data = librosa.feature.mfcc(data, sampling_rate, n_mfcc=40)
    figure, axes = plt.subplots()
    image_mfcc = librosa.display.specshow(mfcc_data, x_axis='time', norm=Normalize(vmin=-40, vmax=40))
    plt.title('MFCC Spectrum Plot of Neutral Sound')
    figure.colorbar(image_mfcc)
    plt.show()


def mel_feature(data, sampling_rate):
    mel_data = librosa.feature.melspectrogram(data, sampling_rate, n_mels=128, fmax=8000)
    mel_data_db = librosa.power_to_db(S=mel_data, ref=np.mean)
    figure, axes = plt.subplots()
    image_mel = librosa.display.specshow(mel_data_db, y_axis='mel', x_axis='time', fmax=8000)
    plt.title('Mel Spectrum Plot of Neutral Sound')
    figure.colorbar(image_mel, ax=axes, format="%+2.0f dB")
    plt.show()


sound_file = 'C:\\Users\\a\\Desktop\\Dataset\\Actor_01\\03-01-02-01-01-01-01.wav'
sound_features(sound_file)
