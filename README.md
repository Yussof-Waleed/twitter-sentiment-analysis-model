# Project Setup Guide

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
