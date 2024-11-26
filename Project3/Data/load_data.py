import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")


def load_mri_data(base_path, batch_size=32, image_size=(128, 128)):
    """
    Loads MRI brain scan data from the specified directory and combines them into one dataset.
    
    Args:
        base_path (str): The base directory containing 'Training' and 'Testing' subfolders.
        image_size (tuple): Target size to resize images (default is 224x224).
        batch_size (int): Batch size for the dataset.

    Returns:
        total_dataset (tf.data.Dataset): Combined dataset with images and labels.
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
        shuffle=True  # Shuffle here to mix data when combining
    )

    # Combine the datasets
    total_dataset = train_ds.concatenate(test_ds)
    
    return total_dataset


def train_test_split(dataset, test_fraction=0.2):
    """
    Splits a tf.data.Dataset into training and testing datasets.
    
    Args:
        dataset (tf.data.Dataset): The dataset to split.
        test_fraction (float): Fraction of the data to use for testing (default 0.2).

    Returns:
        train_ds (tf.data.Dataset): Training dataset.
        test_ds (tf.data.Dataset): Testing dataset.
    """
    # Convert the dataset to a list of (image, label) pairs
    data = list(dataset.as_numpy_iterator())
    images, labels = zip(*data)

    # Stack images and labels
    images = np.vstack(images)
    labels = np.concatenate(labels)

    # Shuffle the data
    total_size = len(labels)
    indices = np.arange(total_size)
    np.random.shuffle(indices)

    images = images[indices]
    labels = labels[indices]

    # Split the data
    split_index = int(total_size * (1 - test_fraction))
    train_images, test_images = images[:split_index], images[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]

    # Convert back to tf.data.Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

    return train_ds, test_ds


if __name__ == '__main__':
    # Define the base directory
    base_path = 'Project3/Data/archive'
    batch_size = 32

    # Load the MRI data
    total_dataset = load_mri_data(base_path, batch_size=batch_size)
    print(f"Total dataset: {len(list(total_dataset))} batches")

    # Split the dataset into training and testing sets
    train_ds, test_ds = train_test_split(total_dataset, test_fraction=0.2)

    print(f"Training dataset: {len(list(train_ds))} batches")
    print(f"Testing dataset: {len(list(test_ds))} batches")

    # Visualize a batch from the training set
    class_names = ['notumor', 'glioma', 'meningioma', 'pituitary']
    
    for image_batch, label_batch in train_ds.take(1):
        plt.figure(figsize=(16, 16))
        for i in range(batch_size):
            plt.subplot(4, 8, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(class_names[label_batch[i]])
            plt.axis('off')
        plt.show()

    for images, labels in train_ds.take(1):
        print("Image pixel values (first image in batch):")
        print(images[0].numpy())  # Convert to NumPy array
        print("Label for the first image in the batch:", labels[0].numpy())



        """DOUBLE CHECK CLASS NAMES"""