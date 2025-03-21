# Project Setup Guide

## Overview

This project focuses on sentiment analysis using deep learning techniques. It includes two main scripts:

### `train.py`
- **Capabilities**: 
  - Downloads and preprocesses the Sentiment140 dataset.
  - Augments the dataset to balance class distribution.
  - Analyzes and visualizes data.
  - Prepares data for training.
  - Builds and trains various neural network models (BiLSTM, Attention GRU, CNN-RNN).
  - Evaluates model performance and saves the trained model.
- **Features**:
  - Enhanced text preprocessing with contraction and emoji handling.
  - Data augmentation techniques (synonym replacement, random deletion, random swap).
  - K-fold cross-validation.
  - Mixed precision training for faster performance.
  - Comprehensive logging and visualization of training and evaluation metrics.

### `test.py`
- **Capabilities**:
  - Loads the latest trained model and its components.
  - Cleans and preprocesses input text.
  - Predicts sentiment of the input text.
  - Generates visual explanations of the prediction.
- **Features**:
  - Interactive mode for real-time sentiment analysis.
  - Batch testing mode for analyzing multiple texts.
  - Visualization of word contributions to the sentiment prediction.

## Prerequisites
- Python 3.6 or higher
- Git (for cloning the repository)
- Internet connection (for downloading dataset and dependencies)

## Getting Started

### 1. Dataset Acquisition
Download the [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) and save it to the project's root directory.

### 2. Environment Setup

**Create a virtual environment:**
```bash
# For Linux/macOS
python3 -m venv env

# For Windows
python -m venv env
```

**Activate the virtual environment:**
```bash
# For Linux/macOS
source env/bin/activate

# For Windows
env\Scripts\activate

# Note: Some Windows installations may have a 'bin' folder instead of 'Scripts'
# If that's the case, use:
env\bin\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset
Rename the downloaded dataset file to `xdata.csv` in the project's root directory.

### 5. Train the Model
```bash
# For Linux/macOS
python3 train.py

# For Windows
python train.py
```

Once training completes, the model will be saved in the `models` directory.
