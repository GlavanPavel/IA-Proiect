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

    return x_normalized, y
