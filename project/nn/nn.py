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
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("No GPU available; train on CPU")

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


random.seed(43)
np.random.seed(43)
torch.manual_seed(43)
model = MyModel().to(device)
print(model)

DATA_DIR, TRAIN_EPOCH = '../data/tfidf_misc/', 200
trainfile, testfile = DATA_DIR + 'tfidf.misc.train.csv', DATA_DIR + 'tfidf.misc.test.csv'
loss_func, scaler = nn.BCELoss(), MinMaxScaler()
lr, bs = 0.01, 128
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

trainset, testset = pd.read_csv(trainfile, header=0).to_numpy(), pd.read_csv(testfile, header=0).to_numpy()

x_tensor, y_tensor = torch.from_numpy(scaler.fit_transform(trainset[:, 1:])).float(), torch.from_numpy(trainset[:, 0]).float().unsqueeze(1)
x_test_tensor, y_test_tensor = torch.from_numpy(scaler.fit_transform(testset[:, 1:])).float(), torch.from_numpy(testset[:, 0]).float().unsqueeze(1)
train_tds, test_tds = TensorDataset(x_tensor, y_tensor), TensorDataset(x_test_tensor, y_test_tensor)
train_dl, test_dl = DataLoader(train_tds, batch_size=bs, num_workers=4, shuffle=True), DataLoader(test_tds, batch_size=1024, num_workers=4)

best_test_accuracy = -0.01
# Training
for epoch in range(TRAIN_EPOCH):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        y_pred = model(xb)
        loss = loss_func(y_pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    scheduler.step()
    
    # Train & Test Accuracy
    model.eval()
    correct, total = 0, 0
    for xb, yb in train_dl:
        xb = xb.to(device)
        y_pred = model(xb)
        predicted = torch.round(y_pred).cpu()
        correct += int((predicted == yb).sum().item())
        total += yb.shape[0]
    train_accuracy = float(correct) / total

    correct, total = 0, 0
    for xb, yb in test_dl:
        xb = xb.to(device)
        y_pred = model(xb)
        predicted = torch.round(y_pred).cpu()
        correct += int((predicted == yb).sum().item())
        total += yb.shape[0]
    test_accuracy = float(correct) / total

    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        torch.save(model.state_dict(), './model_state_dict')

    print("Epoch %d: train_accuracy=%.4f test_accuracy=%.4f" % (epoch+1, train_accuracy, test_accuracy))

print("The best test accuracy = %.4f" % (best_test_accuracy))
