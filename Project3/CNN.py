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
import intel_extension_for_tensorflow as itex; print(itex.__version__)
print(itex.is_enabled())

#Project3/Data/load_data.py
from Data.load_data import load_mri_data, train_test_split
# os.environ["TF_NUM_INTEROP_THREADS"] = "8"  # Adjust based on your CPU cores
# os.environ["TF_NUM_INTRAOP_THREADS"] = "8"




def CNN(input_shape, num_classes, growth_rate=32):
    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Initial layer
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Dense blocks and transition layers
    for num_layers in [6, 12, 24, 16]:  # Standard DenseNet-121 configuration
        x = dense_block(x, num_layers, growth_rate)
        x = transition_layer(x)

    # Global average pooling and output layer
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs)

def dense_block(x, repetition, growth_rate):
    for _ in range(repetition):
        x = bottleneck_layer(x, growth_rate)
    return x

def bottleneck_layer(x, growth_rate):
    skip_connection = x
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(4 * growth_rate, kernel_size=1, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(growth_rate, kernel_size=3, strides=1, padding='same')(x)
    return layers.Concatenate()([x, skip_connection])

def transition_layer(x):
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(x.shape[-1] // 2, kernel_size=1, strides=1, padding='same')(x)
    return layers.AvgPool2D(pool_size=2, strides=2, padding='same')(x)


if __name__ == "__main__":
    base_path = 'Project3/Data/archive'
    batch_size = 32

    # Load the MRI data
    train_set, test_set = load_mri_data(base_path, batch_size=batch_size)
    # print(f"Total dataset: {len(list(total_dataset))} batches")

    # Split the dataset into training and testing sets
    # train_set, test_set = train_test_split(total_dataset, test_fraction=0.2)

    print(f"Training dataset: {len(list(train_set))} batches")
    print(f"Testing dataset: {len(list(test_set))} batches")

    # Visualize a batch from the training set
    class_names = ['notumor', 'glioma', 'meningioma', 'pituitary']


    AUTOTUNE = tf.data.AUTOTUNE

    train_set = train_set.cache().prefetch(buffer_size=AUTOTUNE)
    test_set = test_set.cache().prefetch(buffer_size=AUTOTUNE)


    model = CNN(input_shape=(128, 128, 3), num_classes=4)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=0.001,
    #     decay_steps=10000,
    #     decay_rate=0.9)
    

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001, epsilon=0.05),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    # model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)


    # Training
    history = model.fit(train_set, validation_data=test_set, epochs=20,
                    callbacks=[early_stopping])

    # Visualization
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()