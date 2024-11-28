import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


def load_mri_data(base_path, batch_size=32, image_size=(128, 128)):
    """
    Loads MRI brain scan data from the specified directory and combines them into one dataset.
    
    Args:
        base_path (str): The base directory containing 'Training' and 'Testing' subfolders.
        image_size (tuple): Target size to resize images.
        batch_size (int): Batch size for the dataset.

    Returns:
        total_dataset (torch.utils.data.Dataset): Combined dataset with images and labels.
    """
    # Paths to train and test directories
    train_dir = os.path.join(base_path, 'Training')
    test_dir = os.path.join(base_path, 'Testing')

    print("Batch size: ", batch_size)
    print("Image size: ", image_size)

    # Define transformations (resize images and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    # Load the datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Combine training and testing datasets
    total_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    labels_map = {0: 'glioma tumor', 1: 'meningioma tumor', 2: 'no tumor', 3: 'pituitary tumor'}
    return total_dataset, labels_map


def train_test_split(dataset, test_fraction=0.2, batch_size=32):
    """
    Splits a torch.utils.data.Dataset into training and testing datasets.
    
    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        test_fraction (float): Fraction of the data to use for testing (default 0.2).
        batch_size (int): Batch size for the DataLoader.

    Returns:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
    """
    # Calculate the sizes for training and testing datasets
    total_size = len(dataset)
    test_size = int(total_size * test_fraction)
    train_size = total_size - test_size

    random_seed = 42  # Set random seed for reproducibilityrandom_seed = 42
    generator = torch.Generator().manual_seed(random_seed)

    # Randomly split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size],generator=generator)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def plot_batch(images, labels, labels_map, num_images=9):
    """
    Plots a grid of images with their corresponding labels.

    Args:
        images (Tensor): Batch of images (Shape: [batch_size, channels, height, width]).
        labels (Tensor): Corresponding labels for the images.
        labels_map (dict): Mapping from label indices to class names.
        num_images (int): Number of images to display (default is 9).
    """
    # Limit the number of images to the square root for a clean grid
    grid_size = int(np.sqrt(num_images))
    num_images = grid_size ** 2  # Ensure a perfect square

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        if i >= num_images:
            break
        img = images[i].permute(1, 2, 0).numpy()  # Convert CHW to HWC and to NumPy
        label = labels[i].item()
        ax.imshow(img)
        ax.set_title(f"{labels_map[label]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    base_path = "Project3/Data/archive"  # Adjust to your actual base path
    batch_size = 32
    image_size = (128, 128)

    # Load the dataset
    total_dataset, labels_map = load_mri_data(base_path, batch_size=batch_size, image_size=image_size)

    # Split into training and testing datasets
    train_set, test_set = train_test_split(total_dataset, test_fraction=0.2, batch_size=batch_size)

    # Get a batch from the training set
    train_features, train_labels = next(iter(train_set))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")

    # Plot a grid of images
    plot_batch(train_features, train_labels, labels_map, num_images=9)
