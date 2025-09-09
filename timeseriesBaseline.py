import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, precision_score
from src.helper_function.memory import estimate_memory_training, estimate_memory_inference
import os
from src.genRNN import pretrainedLSTM

print("loading dataset")

datapath = "datasets/desktop_cryptocurrency"
datafilePath = "datasets/desktop_cryptocurrency/{}"
dataDirectory = os.listdir(datapath)

datasets = []
for dataset in dataDirectory:
    df = pd.read_csv(datafilePath.format(dataset))
    datasets.append(df)

# print(datasets)

print("training autoencoder experiment")
encoder = pretrainedLSTM(10, 2, 10)
encoder.init_model()
encoder.getWeights()
encoder.genWeights()