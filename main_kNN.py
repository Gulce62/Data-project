import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from kNN import KNN

emotions_dataframe = pd.read_csv('/Users/a/Desktop/emotions.csv', dtype=str)
features_dataframe = pd.read_csv('/Users/a/Desktop/features.csv')
features_dataframe = features_dataframe.iloc[:, 1:]
emotion_list = emotions_dataframe['0'].values.tolist()
sound_features_list = features_dataframe.values.tolist()

scaling_value = StandardScaler()
scaled_features = sound_features_list
scaled_features = scaling_value.fit_transform(scaled_features)

# TODO make cross validation for finding the percentage of test train validation datasets
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features,
    emotion_list,
    test_size=0.2,
    random_state=60
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=60
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


accuracy = 0
optimal_k = None
optimal_d_type = None
distance_type = ['euclidian distance', 'manhattan distance', 'cosine distance']
for d_type in range(1, 4):
    for k_val in range(1, 11):
        validate_knn = KNN(k=k_val, distance_type=d_type)
        validate_knn.fit(X_train, y_train)
        y_prediction_val = validate_knn.all_prediction(X_val)
        val_accuracy = calculate_accuracy(y_val, y_prediction_val)
        print("Accuracy Percentage of KNN classification for validation data with",
              distance_type[d_type - 1], "and k =", k_val, "is", val_accuracy)
        if val_accuracy >= accuracy:
            optimal_k = k_val
            optimal_d_type = d_type
            accuracy = val_accuracy
print(optimal_k)
print(optimal_d_type)

validate_knn = KNN(k=optimal_k, distance_type=optimal_d_type)
validate_knn.fit(X_train, y_train)
y_prediction_test = validate_knn.all_prediction(X_test)
test_accuracy = calculate_accuracy(y_test, y_prediction_test)
print("Accuracy Percentage of KNN classification for test data with",
      distance_type[optimal_d_type - 1], "and k =", optimal_k, "is", test_accuracy)

confusion_matrix = confusion_matrix(y_test, y_prediction_test)
print(confusion_matrix)

plt.figure(figsize=(10, 6))
ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')
ax.set_title('Emotions Confusion Matrix\n\n')
ax.set_xlabel('\nPredicted Emotion Category')
ax.set_ylabel('Actual Emotion Category ')

ax.xaxis.set_ticklabels(['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'])
ax.yaxis.set_ticklabels(['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'])
plt.show()

# TODO calculate mean squared error
