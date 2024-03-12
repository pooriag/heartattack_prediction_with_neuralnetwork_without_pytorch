import neural_network as nn
import csv_as_dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pnd

def plot_losses(losses):
    plt.figure()

    for i in range(0, len(losses), 1):
        plt.plot(i, losses[i][1], 'ro')
    plt.show()

df = pnd.read_csv('heart.csv')
df['index'] = range(0, len(df))
df.set_index('index', inplace=True)

cad = csv_as_dataset.csv_dataset("output")

norm_df = cad.normalize_data(df)
train_data = norm_df[(norm_df.index % 3 == 0) | (norm_df.index % 3 == 1)]
test_data = norm_df[(norm_df.index % 3 == 2) & (norm_df.index % 2 == 0)]
evaluation_data = norm_df[(norm_df.index % 3 == 2) & (norm_df.index % 2 == 1)]

layers = [[len(norm_df.columns) - 1], [10, nn.relu], [1, nn.sigmoid]]
M = nn.NN_model(layers, 3e-2, 10000, nn.CE)

cad.train(train_data, 40, 20, M)

plot_losses(M.losses)