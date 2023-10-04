import os
import numpy as np
import matplotlib.pyplot as plt
from src.gradient import gradient1

import gzip
import struct

data_folder = os.path.join(os.getcwd(), 'data')

# load compressed MNIST gz files and return numpy arrays
def load_data(filename, label=False):
    with gzip.open(filename) as gz:
        struct.unpack('I', gz.read(4))
        n_items = struct.unpack('>I', gz.read(4))
        if not label:
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            res = np.frombuffer(gz.read(n_items[0] * n_rows * n_cols), dtype=np.uint8)
            res = res.reshape(n_items[0], n_rows * n_cols)
        else:
            res = np.frombuffer(gz.read(n_items[0]), dtype=np.uint8)
            res = res.reshape(n_items[0], 1)
    return res

# note we also shrink the intensity values (X) from 0-255 to 0-1. This helps the model converge faster.
X_train = load_data(os.path.join(data_folder, 'train-images.gz'), False) / 255.0
Y_train = load_data(os.path.join(data_folder, 'train-labels.gz'), True).reshape(-1)

train_len = len(X_train)
X_first = X_train[:2]
Y_first = Y_train[:2]
print(Y_first)

Y_res = []
for y in Y_first:
    mas = [0]*10
    mas[y] = 1
    Y_res.append(mas)


gradient1(X_first, Y_res)