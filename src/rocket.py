import os
os.chdir('/content/drive/MyDrive/fall_detection/src/')

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='paper', style='ticks')

from sklearn.linear_model import RidgeClassifierCV

from sktime.transformations.panel.rocket import Rocket

from utils.utils import load_data


# for saving loop results
results = {}

for _, ds in enumerate(load_data()):
    # load the data
    idx, labels, X_train, X_valid, X_test, y_train, y_valid, y_test = ds

    # swap the axes (1, 2) to match with sktime format
    X_train = np.swapaxes(X_train, 1, 2)
    X_valid = np.swapaxes(X_valid, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)

    # combine training and validation sets
    X_train = np.concatenate((X_train, X_valid), axis=0)
    y_train = np.concatenate((y_train, y_valid), axis=0)

    # get the number of classes
    n_classes = len(np.unique(np.concatenate((y_train, y_valid, y_test), axis=0)))

    # fit the ROCKET transform
    rocket = Rocket(n_jobs=-1, random_state=42)  # by default, ROCKET uses 10,000 kernels
    rocket.fit(X_train)
    X_train_transform = rocket.transform(X_train)

    # fit a ridge classifier
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_transform, y_train)

    # transform the test data
    X_test_transform = rocket.transform(X_test)

    y_pred = classifier.predict(X_test_transform)
    
    results[f'labels_{idx}'] = labels
    results[f'y_true_{idx}'] = y_test
    results[f'y_pred_{idx}'] = y_pred

    # save rocket
    with open(f'rocket/rocket_{idx}.pkl', 'wb') as f:
        pickle.dump(rocket, f)
    # save ridge classifier
    with open(f'rocket/ridge_{idx}.pkl', 'wb') as f:
        pickle.dump(classifier, f)

    # resetting rocket
    rocket.reset()
    # ridge resets automatically when instantiated and fitted again

# Serialize data into file:
with open('rocket/results.pkl', 'wb') as fh:
    pickle.dump(results, fh)