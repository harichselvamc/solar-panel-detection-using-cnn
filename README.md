# Solar Panel Detection using CNN

## Overview
This project focuses on using Convolutional Neural Networks (CNN) to detect solar panels in TIF format images. The CNN model is trained on a dataset containing labeled images, and the trained model can be used to predict whether solar panels are present in new images.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Results](#results)


## Installation
To get started, clone this repository to your local machine:

      git clone https://github.com/harichselvamc/solar-panel-detection-using-cnn.git
      cd solar-panel-detection-using-cnn

Install the required dependencies:
       pip install -r requirements.txt


# Usage
**Load the Data:**
- Place your training images in the `./training/` folder and create a CSV file `labels_training.csv` with image IDs and corresponding labels.
- For testing, place TIF format images in the `./testing/` folder.

**Train the Model:**
- Run the Jupyter notebook `main.ipynb` to train the CNN model.

**Test the Model:**
- Run the Jupyter notebook `testing.ipynb` to use the trained model for predicting solar panels in new images.

# Dataset
The training dataset consists of TIF format images in the `./training/` folder along with labels provided in `labels_training.csv`. The dataset is organized to facilitate easy model training.

# Model Training
The CNN model is built using Keras and TensorFlow. It consists of convolutional layers, batch normalization, and a dense layer for binary classification. The model is trained using the training dataset, and hyperparameters can be adjusted in the `main.ipynb` notebook.

# Results
The model performance can be evaluated by checking the accuracy, precision, recall, and F1-score. Visualizations of true positives, true negatives, false positives, and false negatives are also provided.

