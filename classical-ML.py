import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import sklearn
import scipy
from pathlib import Path

p  = Path('./eye_state_detection/data')
arr = []
for file in sorted(p.iterdir()):
    df = pd.read_csv(file)
    arr.append(df)

train_data = pd.concat(arr[1:], ignore_index= True)
validate_data = arr[0][:arr[0].shape[0]//2]
test_data = arr[0][arr[0].shape[0]//2:].reset_index(drop=True)

features_train_data = train_data.drop(columns = ['eye_state'])
labels_train_data = train_data['eye_state']

features_validate_data = validate_data.drop(columns = ['eye_state'])
labels_validate_data = validate_data['eye_state']

features_test_data = test_data.drop(columns = ['eye_state'])
labels_test_data = test_data['eye_state']


def clean(s, threshold):
    mean = s.mean()
    s = s.copy()
    s[np.abs(s - mean) > threshold] = mean
    return s

def normalize(df):
    return (df - df.mean()) / df.std()

def preprocess(df, threshold):
    df = df.copy()
    df = df.apply(clean, threshold=threshold)
    df = normalize(df)
    return df
    
features_train_data = preprocess(features_train_data, threshold=250)
features_validate_data = preprocess(features_validate_data, threshold=250)
features_test_data = preprocess(features_test_data, threshold=250)
plt.figure()
plt.plot(features_train_data)
plt.savefig("train_plot.png")

plt.figure()
plt.plot(features_validate_data)
plt.savefig("validate_plot.png")


plt.figure()
plt.plot(features_test_data)
plt.savefig("test_plot.png")

model = sklearn.linear_model.LogisticRegression()
model.fit(features_train_data, labels_train_data)
print(model.score(features_validate_data, labels_validate_data))
print(model.score(features_test_data, labels_test_data))

forest = sklearn.ensemble.RandomForestClassifier()
forest.fit(features_train_data, labels_train_data)
print(forest.score(features_validate_data, labels_validate_data))
print(forest.score(features_test_data, labels_test_data))


