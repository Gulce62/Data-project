import numpy as np
import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('/Users/a/Desktop/audio.csv')
df = df.iloc[:, 1:]
df.head()
df.info()
Y = df["0"]
df = df.drop(["0"], axis=1)
X = df.to_numpy()  # np.matrix(df.to_numpy())
y = Y.to_numpy().transpose()  # np.matrix(Y.to_numpy()).transpose()
m, n = X.shape
print(X)
print(y)
normalized_features = preprocessing.normalize(X)
print(normalized_features)

trans = X.T
for i in range(len(trans)):
    mean_val = np.mean(trans[i])
    min_val = np.min(trans[i])
    max_val = np.max(trans[i])
    for j in range(len(trans[i])):
        trans[i][j] = (trans[i][j]-mean_val)/(max_val-min_val)
X = trans.T
