import csv
import numpy as np


def load_data_csv_module(filepath):
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=';')

        # pentru a sari peste header
        next(reader)

        for row in reader:
            float_row = [float(x) for x in row]
            data.append(float_row)

    return np.array(data)

def normalize_data(data):
    x = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)

    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)

    x_normalized = (x - x_min) / (x_max - x_min)

    return x_normalized, y, x_min, x_max

def split_data(data, train_size):
    indices = np.random.permutation(len(data))
    split_point = int(len(data) * train_size)

    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    train_data = data[train_indices]
    test_data = data[test_indices]

    return train_data, test_data

def normalize_data_test(data, x_min, x_max):
    x = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)

    x_normalized = (x - x_min) / (x_max - x_min)

    return x_normalized, y