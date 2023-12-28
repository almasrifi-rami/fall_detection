import os
os.chdir('/content/drive/MyDrive/fall_detection/src/')

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='paper', style='ticks')

from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

import tensorflow as tf
from tensorflow import keras

np.random.seed(42)


# load the dataframe
def load_df(kind=['Wrist'], sensors=['Acc', 'Gyr']):
    """
    Load the DataFrame with the sensor readings and activity labels
    
    Parameters:
        kind (list) -- List of device locations to include
        sensors (list) -- List of sensor names to use
        
    Returns:
        pd.DataFrame with specific sensor readings and their labels
    """
    # read from pickle file
    file_path = '/content/drive/MyDrive/fall_detection/data/FallAllD.pkl'
    df = pd.read_pickle(file_path)
    if kind is None:
        df = df
    else:
        # selecting the device type
        df = df[df['Device'].isin(kind)]
    # setting the index for per subject, activity and trial#
    df = df.set_index(['SubjectID', 'ActivityID', 'TrialNo', 'Device'])
    
    if sensors is None:
        pass
    else:
        # selecting only the valuable sensor values
        df = df.loc[:, sensors]

    return df


# get the features from the dataframe
def sensor_to_same_len(df, include_bar=False):
    """
    Get the features array from the dataframe

    Parameters:
        df (pd.DataFrame) -- DataFrame containing the sensor readings
        include_bar (bool) -- Boolean specifying whether the sensors include barometer

    Returns:
        np.ndarray with shape (n_samples, timesteps, variables)
    """
    # get the number of samples
    n = df.shape[0]
    # maximum length of sensor readings
    MAX_LEN = 4760
    # number of variables of the dataframe
    n_var = len(df.columns)

    # initiate the array of featuers to zeros
    if include_bar == False:
        ucr_x = np.zeros((n, MAX_LEN, n_var * 3))
    else:
        ucr_x = np.zeros((n, MAX_LEN, n_var * 3 - 1))        # add -1 when using all sensors
    
    # loop over the number of samples
    for i in range(n):
        # important variable to keep record of what column we're at
        col_n = 0
        mts = df.iloc[i]
        # loop over the number of variables
        for j in range(n_var):
            # get each stacked sensor reading (x, y, z)
            ts_stacked = mts[j]
            # find the length
            curr_len = ts_stacked.shape[0]
            # get the indices for the array and new for interpolation
            idx = np.arange(curr_len)
            idx_new = np.linspace(0, idx.max(), MAX_LEN)
            # loop over the columns in the array (x, y, z)
            for k in range(ts_stacked.shape[1]):
                # get univariate time series
                ts = ts_stacked[:, k]
                # get an instance of interpolation 1d
                f = interp1d(idx, ts, kind='cubic')
                # apply it to the new index
                ts_new = f(idx_new)
                # replace the 0s with the new ts after interpolation
                ucr_x[i, :, col_n] = ts_new
                # keep record of the number of columns
                col_n += 1

    return ucr_x


# get the labels of activities (fall or daily activity as 1, 0)
def get_labels(df):
    """
    Get the labels array from the dataframe

    Parameters:
        df (pd.DataFrame) -- DataFrame containing the sensor readings

    Returns:
        np.ndarray containing the labels for each sample as fall or not fall
    """
    # get the labels from the multiindex
    labels = df.index.get_level_values(1)

    y = np.where(labels > 100, 1, 0)
    
    return labels, y

# standardize the data
def standardize_data(X_train, X_valid, X_test):
    """
    Standardize the data with mean 0 and std 1

    Parameters:
        X_train (np.ndarray) -- Array of training features
        X_valid (np.ndarray) -- Array of validation features

    Returns:
        tuple (np.ndarray) -- tuple of standardized arrays 
    """
    # get the mean and std across the samples and time steps
    train_mean = np.mean(X_train, axis=(0, 1))
    train_std = np.std(X_train, axis=(0, 1))

    # standardize the train and validation on training mean and std
    x_train_new = (X_train - train_mean) / train_std
    x_valid_new = (X_valid - train_mean) / train_std
    x_test_new = (X_test - train_mean) / train_std

    return x_train_new, x_valid_new, x_test_new


# get unique subject ids
def get_unique_ids(df):
    """
    Retrieve the unique subject IDs from the DataFrame

    Parameters:
        df (pd.DataFrame) -- DataFrame containing the sensor readings

    Returns:
        np.ndarray -- Numpy array containing the unique subject IDs
    """
    # get the subject ids
    subjects = df.index.get_level_values(0)
    unique_ids = np.unique(subjects)

    return unique_ids

# split a dataframe to training and test
def split_test(df, idx):
    """
    Split a DataFrame to training and test parts on test ID

    Parameters:
        df (pd.DataFrame) -- DataFrame containing sensor readings
        idx (int) -- Test subject ID

    Returns:
        tuple (pd.DataFrame) -- Tuple of DataFrames for training and test
    """
    # get the subject ids
    subjects = df.index.get_level_values(0)
    # condition for indexing the test subjects
    mask = np.equal(subjects, idx)

    # index into the dataframe to split to training and test
    df_train = df.loc[~mask]
    df_test = df.loc[mask]

    return df_train, df_test


def load_data(kind=['Wrist'], sensors=['Acc', 'Gyr'], include_bar=False):
    """
    Retrieve the dataset split into Training, validation and Testing sets using a leave-one-out approach

    Parameters:
        kind (list) -- List of device locations to include
        sensors (list) -- List of sensor names to use
        include_bar (bool) -- Boolean specifying whether the sensors include barometer

    Returns:
        tuple (np.ndarray) -- Numpy arrays relating to each set
    """
    # load the dataframe
    df = load_df(kind=kind, sensors=sensors)

    # get the test ids
    test_ids = get_unique_ids(df)

    # loop over the test ids
    for idx in test_ids:
        # split to trainig and test
        df_train, df_test = split_test(df, idx)

        # split into X and y
        X_ = sensor_to_same_len(df_train, include_bar=include_bar)
        _, y_ = get_labels(df_train)

        # split into training, validation and test
        idx_train, idx_valid, y_train, y_valid = train_test_split(
            np.arange(X_.shape[0]), y_, test_size=0.2,
            shuffle=True, stratify=y_, random_state=42
        )
        X_train, X_valid = X_[idx_train], X_[idx_valid]

        # split into X and y
        X_test = sensor_to_same_len(df_test)
        labels, y_test = get_labels(df_test)

        # standardize the data
        X_train, X_valid, X_test = standardize_data(X_train, X_valid, X_test)

        yield idx, labels, X_train, X_valid, X_test, y_train, y_valid, y_test


### data analysis
# open the results files
def open_results(file_path):
    with open(file_path, 'rb') as fh:
        results = pickle.load(fh)

    return results


# get the results from the files
def get_results(results_dict):
    labels = {}
    Y_TRUE = {}
    Y_PRED = {}

    for k, v in results_dict.items():
        if k.startswith('labels_'):
            labels[k[k.find('_') + 1:]] = v
        elif k.startswith('y_true_'):
            Y_TRUE[k[k.find('true_') + 5:]] = v
        elif k.startswith('y_pred_'):
            Y_PRED[k[k.find('pred_') + 5:]] = v

    return labels, Y_TRUE, Y_PRED


def compare_results(Y_TRUE, Y_PRED, scoring='accuracy'):
    results = {}

    for k in Y_TRUE.keys():
        y_true = Y_TRUE[k]
        y_pred = Y_PRED[k]
        if scoring == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif scoring == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_pred)
        elif scoring == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif scoring == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif scoring == 'specificity':
            score = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        elif scoring == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        results[k] = results.get(k, score)

    return results


def analyze_metrics(files, scoring):
    """
    Function to analyze various metrics of the results files

    Parameters:
        files (dict) -- dictionary of labels and files to analyze
        df (pd.DataFrame) -- DataFrame object that includes the original data

    Returns:
        pd.DataFrame -- DataFrame with the results analysis
    """
    report = {}
    for k, v in files.items():
        res = open_results(v)
        _, Y_TRUE, Y_PRED = get_results(res)
        metric_by_subject = compare_results(Y_TRUE, Y_PRED, scoring=scoring)
        report[k] = report.get(k, metric_by_subject)
    report_df = pd.DataFrame(report) * 100
    report_df.index.name = 'Subject_ID'

    return report_df


def replace_func(value):
    replacement_dict = {
        101: 'Fall F, walking, trip',
        102: 'Fall F, walking, trip, rec',
        103: 'Fall F, walking, slip',
        104: 'Fall F, walking, slip, rec',
        105: 'Fall F, walking, slip, rot',
        106: 'Fall F, walking, slip, rot, rec',
        107: 'Fall B, walking, slip',
        108: 'Fall B, walking, slip, rec',
        109: 'Fall B, walking, slip, rot',
        110: 'Fall B, walking, slip, rot, rec',
        111: 'Fall F, walking, syncope',
        112: 'Fall B, walking, syncope',
        113: 'Fall L, walking, syncope',
        114: 'Fall F, walking, syncope, attempt protection',
        115: 'Fall F, attempt sit/lie down',
        116: 'Fall F, attempt sit/lie down, rec',
        117: 'Fall B, attempt sit/lie down',
        118: 'Fall B, attempt sit/lie down, rec',
        119: 'Fall L, attempt sit/lie down',
        120: 'Fall L, attempt sit/lie down, rec',
        121: 'Fall F, jog, trip',
        122: 'Fall F, jog, trip, rec',
        123: 'Fall F, jog, slip',
        124: 'Fall F, jog, slip, rec',
        125: 'Fall F, jog, slip, rot',
        126: 'Fall F, jog, slip, rot, rec',
        127: 'Fall L, bed',
        128: 'Fall L, bed, rec',
        129: 'Fall F, chair, syncope',
        130: 'Fall B, chair, syncope',
        131: 'Fall L, chair, syncope',
        132: 'Fall F, syncope',
        133: 'Fall B, syncope',
        134: 'Fall L, syncope',
        135: 'Fall V, syncope, ending sitting',
        1: 'Start clap hands',
        2: 'Clap hands',
        3: 'Stop clap hands',
        4: 'Clap hands one time',
        5: 'Start wave hands',
        6: 'Wave hands',
        7: 'Stop wave hands',
        8: 'Raise hands up',
        9: 'Move hands down',
        10: 'Move hands up then down immediately',
        11: 'Hand shake',
        12: 'Beat table with hand',
        13: 'Sit down',
        14: 'Stand up',
        15: 'Fail to stand up',
        16: 'Lie down on bed',
        17: 'Change position in bed',
        18: 'Rise up from bed',
        19: 'Start walking',
        20: 'Walk slowly',
        21: 'Stop walking',
        22: 'Walk quickly',
        23: 'Stumble while walking, no fall',
        24: 'Jog slowly',
        25: 'Jog quickly',
        26: 'Jump slightly',
        27: 'Jump strongly',
        28: 'Bend down then rise up',
        29: 'Start going upstairs',
        30: 'Going upstairs',
        31: 'Stop going upstairs',
        32: 'Start going downstairs',
        33: 'Going downstairs',
        34: 'Stop going downstairs',
        35: 'Going upstairs quickly',
        36: 'Going downstairs quickly',
        37: 'Start ascending using a lift',
        38: 'Stop ascending using a lift',
        39: 'Start descending using a lift',
        40: 'Stop descending using a lift',
        41: 'Standing in moving bus',
        42: 'Sitting in moving bus',
        43: 'Start jogging',
        44: 'Stop jogging'
    }

    return replacement_dict.get(value, value)

v_replace_func = np.vectorize(replace_func)


def analyze_errors(labels, Y_TRUE, Y_PRED):
    activities_fn = []
    activities_fp = []

    for k in Y_TRUE.keys():
        y_true = Y_TRUE[k]
        y_pred = Y_PRED[k]
        activities = labels[k]
        labeled_activities = v_replace_func(activities)

        fn = np.where(np.logical_and(np.equal(y_true, 1), np.equal(y_pred, 0)), True, False)
        fp = np.where(np.logical_and(np.equal(y_true, 0), np.equal(y_pred, 1)), True, False)

        activities_fn.extend(labeled_activities[fn])
        activities_fp.extend(labeled_activities[fp])

    return activities_fn, activities_fp


def model_confusion(file_path, model_name):
    labels, Y_TRUE, Y_PRED = get_results(open_results(file_path))
        
    # concatenate all predictions and labels
    y_pred = np.concatenate(list(Y_PRED.values()), axis=0)
    y_true  = np.concatenate(list(Y_TRUE.values()), axis=0)

    # get the confusion matrix
    confusion_mtx = tf.math.confusion_matrix(
        y_true, y_pred).numpy()

    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_mtx,
                annot=True, fmt='g')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()