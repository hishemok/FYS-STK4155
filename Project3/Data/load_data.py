import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")



def load_mri_data(base_path, batch_size=32, image_size=(224, 224)):
    """
    Loads MRI brain scan data from the specified directory.
    
    Args:
        base_path (str): The base directory containing 'Train' and 'Test' subfolders.
        image_size (tuple): Target size to resize images (default is 224x224).
        batch_size (int): Batch size for the dataset.

    Returns:
        train_ds (tf.data.Dataset): The training dataset.
        test_ds (tf.data.Dataset): The testing dataset.
    """
    # Paths to train and test directories
    train_dir = os.path.join(base_path, 'Training')
    test_dir = os.path.join(base_path, 'Testing')

    print("Batch size: ", batch_size)
    print("Image size: ", image_size)
    # Load the training dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        interpolation='nearest',
        label_mode='int',  # Integer labels
        class_names=['notumor', 'glioma', 'meningioma', 'pituitary'],
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Load the testing dataset
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='int',  # Integer labels
        interpolation='nearest',
        class_names=['notumor', 'glioma', 'meningioma', 'pituitary'],
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    return train_ds, test_ds

if __name__ == '__main__':
    # Define the base directory
    base_path = 'Project3/Data/archive'
    batch_size = 32
    # Load the MRI data
    train_ds, test_ds = load_mri_data(base_path,batch_size=batch_size)
    print("Len train ds: ",len(train_ds))
    print("Train ds take 1: ", train_ds.take(1))

    class_names = train_ds.class_names
    
    for image, label in train_ds.take(1):
        plt.figure(figsize=(16, 16))
        for i in range(batch_size):
            plt.subplot(4, 8, i+1)
            plt.imshow(image[i])
            plt.title(class_names[label[i]])
            plt.axis('off')
        plt.show()