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
M = nn.NN_model(layers, 3e-3, 100, nn.CE)

cad.train(train_data, 20, 20, M)

train_accuracy_batch = csv_as_dataset.random_sample(train_data, 100)
c = 0
for i in range(len(train_accuracy_batch)):
    out = train_accuracy_batch["output"].iloc[i]

    x = train_accuracy_batch[train_accuracy_batch.columns[:-1]]
    X = np.array(x.values.tolist()[i]).T
    X = X.reshape(-1, 1)
    A, caches = M.forward(X)
    if(abs(out - A) < 0.5):
        c += 1

print(f"train accuracy:{c}")
#################
test_accuracy_batch = csv_as_dataset.random_sample(test_data, 100)

c = 0
for i in range(len(test_accuracy_batch)):
    out = test_accuracy_batch["output"].iloc[i]

    x = test_accuracy_batch[test_accuracy_batch.columns[:-1]]
    X = np.array(x.values.tolist()[i]).T
    X = X.reshape(-1, 1)
    A, caches = M.forward(X)
    if(abs(out - A) < 0.5):
        c += 1
print(f"test accuracy:{c}")

###########
test_sample = csv_as_dataset.random_sample(test_data, 20)
for i in range(len(test_sample)):
    print(f'actual value{test_sample["output"].iloc[i]}')

    x = test_sample[test_sample.columns[:-1]]
    X = np.array(x.values.tolist()[i]).T
    X = X.reshape(-1, 1)
    A, caches = M.forward(X)
    print(f'prediction{A}')
    print("..............................")

plot_losses(M.losses)