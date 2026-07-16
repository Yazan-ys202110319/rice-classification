# Rice Classification

This project is a notebook-based binary classification workflow that predicts the class of rice grains from their shape and texture measurements. The model is implemented in PyTorch and trained on the dataset in `Data/riceClassification.csv`.

## Overview

The notebook walks through the full machine learning pipeline:

- loading and cleaning the dataset
- dropping missing values and removing the `id` column
- normalizing the numeric features
- splitting the data into training, validation, and test sets
- building a custom PyTorch `Dataset` and `DataLoader`
- training a small feedforward neural network
- evaluating the model with accuracy and loss
- running a simple inference example on a custom rice sample

## Dataset

The dataset contains rice grain measurements such as:

- Area
- MajorAxisLength
- MinorAxisLength
- Eccentricity
- ConvexArea
- EquivDiameter
- Extent
- Perimeter
- Roundness
- AspectRation

The target column is `Class`, which makes this a binary classification problem.

## Model

The notebook uses a simple neural network built with PyTorch:

- one hidden layer
- sigmoid output for binary classification
- binary cross-entropy loss
- Adam optimizer

## Requirements

Install the dependencies listed in `requirements.txt` before running the notebook.

## How to Run

1. Create and activate a Python virtual environment.
2. Install the dependencies:

	```bash
	pip install -r requirements.txt
	```

3. Open `main.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
4. Run the cells from top to bottom.

## Project Structure

- `main.ipynb` - notebook with preprocessing, training, evaluation, and inference
- `Data/riceClassification.csv` - dataset used for training and evaluation
- `requirements.txt` - Python dependencies for the project

## Notes

- The notebook automatically uses CUDA if PyTorch detects a compatible GPU, otherwise it falls back to CPU.
- The inference example at the end of the notebook shows how to normalize custom input values before prediction.