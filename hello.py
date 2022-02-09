import streamlit as st
import re
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from PIL import Image
from sklearn.preprocessing import MinMaxScaler

from hparams import num_classes, input_size, num_layers, hidden_size
from model import LSTM
from train import train

numbers = st.text_input("Please enter sequence at lest 25 sequence needs for LSTM predict")

if numbers:
    try:
        series = [float(i) for i in numbers.split(",")]
        original = series.copy()
        training_set = np.array(series)
        training_set = np.expand_dims(training_set, axis=1)

        sc = MinMaxScaler()
        training_data = sc.fit_transform(training_set)

        train(training_data)

        model = LSTM(input_size, hidden_size, num_classes, num_layers)
        model.load_state_dict(torch.load('weight/weight.pth'))
        model.eval()

        n_days = 11
        res = []
        data = training_data.tolist()

        for i in range(1, n_days):
            test = torch.Tensor(np.array(data[-12:]))

            predict = model(test.unsqueeze(0)).data.numpy()
            res.append(predict.item())

            data.append(predict.tolist()[0])

        result = np.array(res)
        final = sc.inverse_transform(np.expand_dims(result, axis=1))
        output = np.squeeze(final,axis=1).tolist()

        st.write(str(list(map(lambda x: round(x,2), output))))

        fig = plt.figure(figsize=(8, 5))
        value_len = len(original)
        list1 = [i for i in range(1, value_len + 1)]
        list2 = [i for i in range(value_len + 1, value_len + 11)]
        plt.plot(list1, original)
        plt.plot(list2, output)
        fig.savefig('result.png')

        image = Image.open('result.png')
        st.image(image, caption='Result')


    except Exception as e:
        st.write(e)
