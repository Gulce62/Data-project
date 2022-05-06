import librosa
import numpy as np
import soundfile as sf
import librosa.display
import os


# Modality (01 = full-AV, 02 = video-only, 03 = audio-only) --> Always 03
# Vocal channel (01 = speech, 02 = song) --> Always 01
# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
# Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
# Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
# Repetition (01 = 1st repetition, 02 = 2nd repetition).
# Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).


def sound_features(sound_path):
    data, sampling_rate = sf.read(sound_path, dtype='float32')
    if len(data.shape) != 1:
        return 1
    else:
        chroma = chroma_feature(data, sampling_rate)
        mfcc = mfcc_feature(data, sampling_rate)
        mel = mel_feature(data, sampling_rate)
        contrast = contrast_feature(data, sampling_rate)
        tonnetz = tonnetz_feature(data, sampling_rate)
        sound_feature_matrix = np.concatenate((chroma, mel, mfcc, contrast, tonnetz), axis=0, out=None)
        return sound_feature_matrix


def chroma_feature(data, sampling_rate):
    stft_data = librosa.stft(data)
    stft_data_abs = np.abs(stft_data)
    chroma_data = librosa.feature.chroma_stft(S=stft_data_abs, sr=sampling_rate)
    chroma_mean = np.mean(chroma_data.T, axis=0)
    return chroma_mean


def mfcc_feature(data, sampling_rate):
    mfcc_data = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40)
    mfcc_mean = np.mean(mfcc_data.T, axis=0)
    return mfcc_mean


def mel_feature(data, sampling_rate):
    mel_data = librosa.feature.melspectrogram(y=data, sr=sampling_rate, n_mels=128, fmax=8000)
    mel_mean = np.mean(mel_data.T, axis=0)
    return mel_mean


def contrast_feature(data, sampling_rate):
    stft_data = librosa.stft(data)
    stft_data_abs = np.abs(stft_data)
    contrast_data = librosa.feature.spectral_contrast(S=stft_data_abs, sr=sampling_rate)
    contrast_mean = np.mean(contrast_data.T, axis=0)
    return contrast_mean


def tonnetz_feature(data, sampling_rate):
    harmonic_data = librosa.effects.harmonic(data)
    tonnetz_data = librosa.feature.tonnetz(y=harmonic_data, sr=sampling_rate)
    tonnetz_mean = np.mean(tonnetz_data.T, axis=0)
    return tonnetz_mean


def get_feature_lists(audio_main_path):
    count = 0
    is_error = False
    emotion_list = []
    sound_features_list = []
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
            if isinstance(sound_feature, int):
                count += 1
                print('error')
                is_error = True
            else:
                if is_error:
                    emotion_list.append(emotion)
                    sound_features_list.append(sound_feature)
                    is_error = False
                print(audio_file)
                emotion_list.append(emotion)
                sound_features_list.append(sound_feature)
        print(actor)
    print(count)
    return emotion_list, sound_features_list
