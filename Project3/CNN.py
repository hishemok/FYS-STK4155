import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import warnings
warnings.filterwarnings("ignore")
from tensorflow import keras
from tensorflow.keras import layers, models, metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Project3/Data/load_data.py
from Data.load_data import load_mri_data




if __name__ == "__main__":
    base_path = 'Project3/Data/archive'
    batch_size = 32
    # Load the MRI data
    train_ds, test_ds = load_mri_data(base_path,batch_size=batch_size)
    
    num_batches_train = len(train_ds)
    num_batches_test = len(test_ds)
    batch_size_train = train_ds.element_spec[0].shape[0]
    batch_size_test = test_ds.element_spec[0].shape[0]

    print("Number of batches in training set: ", num_batches_train)
    print("Number of batches in testing set: ", num_batches_test)
    print("Batch size in training set: ", batch_size_train)
    print("Batch size in testing set: ", batch_size_test) 

        # Define a simple CNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 classes
    ])

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model directly using tf.data.Dataset
    model.fit(train_ds, validation_data=test_ds, epochs=10)

