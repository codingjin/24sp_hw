import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")
    #print("GPU is available")
else:
    device = torch.device("cpu")
    #print("No GPU available; train on CPU")

n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_output = 12000, 5000, 1000, 100, 1
class MyModel(nn.Module):
    def __init__(self, dropout_p=0.5, input_dim=10267):
        super(MyModel, self).__init__()
        self.dropout_p = dropout_p
        self.input_dim = input_dim

        global n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_output
        self.model = nn.Sequential(
            nn.Linear(input_dim, n_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(n_hidden2, n_hidden3),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(n_hidden3, n_hidden4),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(n_hidden4, n_output),
            nn.Sigmoid()
        )

        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)


    def forward(self, inputs):
        return self.model(inputs)

scaler = MinMaxScaler()
DATA_DIR = '../data/tfidf_misc/'
evalfile = DATA_DIR + 'tfidf.misc.eval.csv'
evalset = pd.read_csv(evalfile, header=0).to_numpy()
#x_tensor, y_tensor = torch.from_numpy(scaler.fit_transform(evalset[:, 1:])).float(), torch.from_numpy(scaler.fit_transform(evalset[:, 0])).float().unsqueeze(1)
x_tensor  = torch.from_numpy(scaler.fit_transform(evalset[:, 1:])).float()
x_tensor = x_tensor.to(device)


random.seed(43)
np.random.seed(43)
torch.manual_seed(43)
model = MyModel()
model.load_state_dict(torch.load('model_state_dict'))
model = model.to(device)
#print(model)
model.eval()

index = 0
for x in x_tensor:
    label = int(torch.round(model(x)).cpu().item())
    print("%d,%d" % (index, label))
    index += 1
