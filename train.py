import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler

from hparams import num_classes, input_size, hidden_size, num_epochs, num_layers, learning_rate
from model import LSTM


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data) - seq_length - 1):
        _x = data[i:(i + seq_length)]
        _y = data[i + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


def train(training_data, max_len):
    seq_length = 12
    x, y = sliding_windows(training_data, seq_length)

    train_size = int(len(y) * 0.50)
    test_size = len(y) - train_size

    dataX = Variable(torch.Tensor(np.array(x)))
    dataY = Variable(torch.Tensor(np.array(y)))

    trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

    testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
    testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))

    best_valid_loss = 2

    lstm = LSTM(input_size, hidden_size, num_classes, num_layers, max_len)

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm(dataX)
        optimizer.zero_grad()
        loss = criterion(outputs, dataY)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

            with torch.no_grad():
                val = lstm(testX)
                val_loss = criterion(val, testY).item()
                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    print("Weight saved! ", val_loss)
                    torch.save(lstm.state_dict(), './weight/weight.pth')


if __name__ == '__main__':
    training_set = np.array(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    training_set = np.expand_dims(training_set, axis=1)

    sc = MinMaxScaler()
    training_data = sc.fit_transform(training_set)

    train(training_data, len(training_set))
