import pandas as pd
from six import StringIO

from soundFeatures import get_feature_lists

import warnings

warnings.filterwarnings('ignore')

audio_main_path = "/Users/a/Desktop/Dataset/"
[emotion_list, sound_features_list] = get_feature_lists(audio_main_path)

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
emotion_dataframe.to_csv('/Users/a/Desktop/emotions2.csv')
# emotion_dataframe = emotion_dataframe.replace(emotion_dict)
features_dataframe = pd.DataFrame(sound_features_list)
features_dataframe.to_csv('/Users/a/Desktop/features2.csv')
main_dataframe = pd.concat([emotion_dataframe, features_dataframe], axis=1)
main_dataframe.to_csv('/Users/a/Desktop/data2.csv')



