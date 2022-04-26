import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from soundFeatures import get_feature_lists
from kNN import KNN

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

# TODO make cross validation for finding the percentage of test train validation datasets
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features,
    emotion_list,
    test_size=0.1,
    random_state=1
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=1/9,
    random_state=1
)

# TODO look for reshape necessity

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


# distance_type = ['euclidian distance', 'manhattan distance', 'cosine distance']
# for d_type in range(1, 4):
#     classify_knn = KNN(k=3, distance_type=d_type)
#     classify_knn.fit(X_train, y_train)
#     y_prediction_train = classify_knn.all_prediction(X_train)
#     y_prediction_test = classify_knn.all_prediction(X_test)
#     print("Accuracy Percentage of KNN classification for train data with",
#           distance_type[d_type - 1], "is", calculate_accuracy(y_train, y_prediction_train))
#     print("Accuracy Percentage of KNN classification for test data with",
#           distance_type[d_type - 1], "is", calculate_accuracy(y_test, y_prediction_test))

accuracy = 0
optimal_k = None
optimal_d_type = None
distance_type = ['euclidian distance', 'manhattan distance', 'cosine distance']
for d_type in range(1, 4):
    for k_val in range(1, 10):
        validate_knn = KNN(k=k_val, distance_type=d_type)
        validate_knn.fit(X_train, y_train)
        y_prediction_val = validate_knn.all_prediction(X_val)
        val_accuracy = calculate_accuracy(y_val, y_prediction_val)
        print("Accuracy Percentage of KNN classification for test data with",
              distance_type[d_type - 1], "and k =", k_val, "is", val_accuracy)
        if val_accuracy >= accuracy:
            accuracy = val_accuracy
            optimal_k = k_val
            optimal_d_type = d_type
print(optimal_k)
print(optimal_d_type)

# TODO ask if the accuracy same
# TODO create confusion matrix
# TODO calculate mean squared error
