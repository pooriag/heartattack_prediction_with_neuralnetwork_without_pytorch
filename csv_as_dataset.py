import random
import numpy as np
import pandas as pnd

class csv_dataset():
    def __init__(self, output_col):
        self.output_col = output_col
    def normalize_data(self, df):
        mean = df.drop(self.output_col, axis=1).mean()
        std = df.drop(self.output_col, axis=1).std()
        df[df.drop(self.output_col, axis=1).columns] = (df.drop(self.output_col, axis=1) - mean) / std
        return df

    def train(self, df, iteration, count, M):
        for i in range(iteration):
            samp = random_sample(df, count)
            x = samp.drop(self.output_col, axis=1)
            y = samp[self.output_col]

            X = np.array(x.values.tolist()).T
            Y = np.array(y.values.tolist()).T
            Y = Y.reshape(1, Y.shape[0])

            M.train(X, Y)

def random_sample(df, count):
    return df.iloc[[random.randint(0, len(df) - 1) for i in range(count)]]