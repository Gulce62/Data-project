import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from soundFeatures import get_feature_lists
from svm import SVM

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
emotion_dataframe = emotion_dataframe.replace(emotion_dict)
features_dataframe = pd.DataFrame(sound_features_list)
main_dataframe = pd.concat([emotion_dataframe, features_dataframe], axis=1)

main_dataframe.to_csv('/Users/a/Desktop/audio.csv')

scaling_value = StandardScaler()
scaled_features = sound_features_list
scaled_features = scaling_value.fit_transform(scaled_features)
emotion_features = np.array(emotion_list)


X_train, X_test, y_train, y_test = train_test_split(
    scaled_features,
    emotion_list,
    test_size=0.1,
    random_state=1
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=1 / 9,
    random_state=1
)

print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("X_val shape: {}".format(X_val.shape))
print("y_train shape: {}".format(np.array(y_train).shape))
print("y_test shape: {}".format(np.array(y_test).shape))
print("y_val shape: {}".format(np.array(y_val).shape))


def calculate_accuracy(true_value, prediction_value):
    size = len(true_value)
    sum_val = 0
    for item in range(size):
        if true_value[item] == prediction_value[item]:
            sum_val += 1
    print(sum_val, 'values are predicted true from', size)
    accuracy_percentage = sum_val / size * 100
    return accuracy_percentage


accuracy = 0

cList = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
kernelList = ["linear", "polynomial", "gaussian"]

svmClassifier = SVM(kernel=kernelList[0], regularization_parameter_C=cList[5])

# disgust training data (1)
yDisgustTrain = np.where(y_train == "disgust", 1, -1)
svmClassifier.fit(X_train, yDisgustTrain)
s = svmClassifier.emotion_predict(X_test)
# prediction making
yTest = np.where(s == 1, "disgust", s)
# disgust test data

print("SVM classification accuracy", " kernel parameter=",
      kernelList[0], " C parameter=", cList[5],
      calculate_accuracy(y_test, yTest))

