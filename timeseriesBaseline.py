import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, precision_score
from src.helper_function.memory import estimate_memory_training, estimate_memory_inference

print("loading dataset")
dataframe  = pd.read_csv()

num_dp, num_input = dataframe.shape

test_index = np.random.choice(range(num_dp), size=int(num_dp*0.2))
train_index = [i for i in range(num_dp) if i not in test_index]

