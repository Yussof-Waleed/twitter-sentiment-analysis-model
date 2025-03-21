1. Download the dataset to the main directory:
[Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

2. Create a virtual environment:
• Linux: python3 -m venv env  
• Windows: python -m venv env  
• macOS: python3 -m venv env  

3. Activate the virtual environment:
• Linux/macOS: source env/bin/activate  
• Windows: env\Scripts\activate  

4. Install project dependencies:
```bash
pip install -r requirements.txt
```

5. Rename the downloaded dataset file to:
xdata.csv

6. Train the model:
```bash
python train.py
```
(For Linux/macOS, use python3 instead of python if necessary)
