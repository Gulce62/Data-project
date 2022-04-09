import pandas as pd
import matplotlib.pyplot as plt
import os
import soundFeatures

import warnings

warnings.filterwarnings('ignore')

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
        sound_feature = soundFeatures.sound_features(file_path)
        if isinstance(sound_feature, int):
            print('error')
        else:
            print(audio_file)
            emotion_list.append(emotion)
            intensity_list.append(emotion_intensity)
            statement_list.append(statement)
            repetition_list.append(repetition)
            actor_list.append(actor_number)
            gender_list.append(int(actor_number) % 2)
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
main_dataframe.to_csv('/Users/a/Desktop/all_properties.csv')

data = pd.read_csv('/Users/a/Desktop/all_properties.csv')
data.rename(columns={'0': 'actors', '0.1': 'emotions', '0.2': 'intensity',
                     '0.3': 'statement', '0.4': 'repetition', '0.5': 'gender',
                     '0.6': 'path'}, inplace=True)
print(data)
data.emotions.value_counts().plot(kind='bar')
data.to_csv('/Users/a/Desktop/all_properties1.csv')
plt.show()
