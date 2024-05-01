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

n_hidden1 = 12000
n_hidden2 = 5000
n_hidden3 = 1000
n_hidden4 = 100
n_output = 1

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

#model = MyModel()
#print(model)

random.seed(43)
np.random.seed(43)
torch.manual_seed(43)

DATA_DIR, CV_EPOCH = '../data/tfidf_misc/', 10

cv_dataset = []
for i in range(5):
    cv_dataset.append(pd.read_csv(DATA_DIR + f'CVfolders/fold{i}.csv', header=0).to_numpy())

loss_func, scaler = nn.BCELoss(), MinMaxScaler()
best_lr, best_bs, best_accuracy = -0.01, -0.01, -0.01
LR_LIST, BATCHSIZE_LIST = [0.1, 0.01, 0.001, 0.0001], [128, 64, 32]
for lr in LR_LIST:
    for bs in BATCHSIZE_LIST:
        validation_accuracies = []
        for i in range(5):
            cv_trainset = np.concatenate((cv_dataset[(i+1)%5], cv_dataset[(i+2)%5]), axis=0)
            cv_trainset = np.concatenate((cv_trainset, cv_dataset[(i+3)%5]), axis=0)
            cv_trainset = np.concatenate((cv_trainset, cv_dataset[(i+4)%5]), axis=0)
            cv_validationset = cv_dataset[i]

            x_tensor, y_tensor = torch.from_numpy(scaler.fit_transform(cv_trainset[:, 1:])).float(), torch.from_numpy(cv_trainset[:, 0]).float().unsqueeze(1)
            x_validation_tensor, y_validation_tensor = torch.from_numpy(scaler.fit_transform(cv_validationset[:, 1:])).float(), torch.from_numpy(cv_validationset[:, 0]).float().unsqueeze(1)
            train_tds, validation_tds = TensorDataset(x_tensor, y_tensor), TensorDataset(x_validation_tensor, y_validation_tensor)
            train_dl, validation_dl = DataLoader(train_tds, batch_size=bs, num_workers=10, shuffle=True), DataLoader(validation_tds, batch_size=1024, num_workers=10)
            
            random.seed(43)
            np.random.seed(43)
            torch.manual_seed(43)
            model = MyModel().to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
            
            # Training
            for epoch in range(CV_EPOCH):
                model.train()
                for xb, yb in train_dl:
                    xb, yb = xb.to(device), yb.to(device)
                    y_pred = model(xb)
                    loss = loss_func(y_pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
            
            # Validation
            model.eval()
            correct, total = 0, 0
            for xb, yb in validation_dl:
                xb = xb.to(device)
                y_pred = model(xb)
                predicted = torch.round(y_pred).cpu()
                correct += int((predicted == yb).sum().item())
                total += yb.shape[0]
            current_validation_accuracy = float(correct) / total
            validation_accuracies.append(current_validation_accuracy)
            #print("current_validation_accuracy=%f" % (current_validation_accuracy))
        validation_accuracy = np.mean(validation_accuracies)
        if validation_accuracy > best_accuracy:
            best_lr, best_bs, best_accuracy = lr, bs, validation_accuracy
        
        print("lr=%.4f bs=%d cv_accuracy=%.4f" % (lr, bs, validation_accuracy))

print("The best hyper-parameter is: lr=%.4f bs=%d cv_accuracy=%.4f" % (best_lr, best_bs, best_accuracy))
