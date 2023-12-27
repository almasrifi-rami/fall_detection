import os
os.chdir('/content/drive/MyDrive/fall_detection/src/')

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='paper', style='ticks')

import tensorflow as tf
from tensorflow import keras

from utils.utils import load_data
from utils.models import resnet_model

tf.random.set_seed(42)


# for saving loop results
results = {}

for _, ds in enumerate(load_data()):
    # load the data
    idx, labels, X_train, X_valid, X_test, y_train, y_valid, y_test = ds
    # get the number of classes
    n_classes = len(np.unique(np.concatenate((y_train, y_valid, y_test), axis=0)))

    # instantiate the model
    model = resnet_model(input_shape=X_train.shape[1: ], nb_classes=n_classes)

    epochs = 1500
    batch_size = 64

    # callbacks for saving the best model, early stopping and reducing lr
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            f'resnet/{idx}.h5', save_best_only=True, monitor='val_loss'
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1),
    ]

    # compile the model with adam optimizer, loss function and f1_score metric
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    # fit the model 
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(X_valid, y_valid),
        verbose=1,
    )

    # reset the model 
    keras.backend.clear_session()

    # evaluate the model on the validation set
    model = keras.models.load_model(f'resnet/{idx}.h5')

    # predict on test values
    y_pred = np.argmax(model.predict(X_test), axis=1)
    
    results[f'labels_{idx}'] = labels
    results[f'y_true_{idx}'] = y_test
    results[f'y_pred_{idx}'] = y_pred
    
# Serialize data into file:
with open('resnet/results.pkl', 'wb') as fh:
    pickle.dump(results, fh)