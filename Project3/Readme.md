### Project 3

## Description
Convolutional Neural Networks for classifying MRI images of brain tumors.

## Data
- Archive: Contains Training and Test folders, of MRI images
- Results: Folder contains different results gathered from Src/CNN_class2.py
- load_data.py: Pulls data and formats it from Archive

## Src
- CNN_class2.py: Class containing a selection of models. Contains usage and Doctypes, and complete method
- CNN_pt.py: Simple first draft of densenet classifier using pytorch
- gradientboosting_gd.ipynb: Implementation of gradient boosting, and hyperparameter tuning. Implemented on pre-trained CNNs from pytorch library
- random_forests.ipynb: Implementation of random forests, and hyperparameter tuning. Implemented on pre-trained CNNs from pytorch library


## CNN & PCA

CNN_Analysis.ipynb - Classification and Analysis with Pytorch

- Dataset pre processing and visualization
- CNN implementation using GPUs
- Analysis of the accuracies and convergence
- Learning rate tunning

PCA_Analysis.ipynb - Reduction of components

- Implementation of the PCA to reduce images to 1000 main components
- Use Logistic Regression for classification
- Use Random Forest to classification
- Use FFNN for classification
