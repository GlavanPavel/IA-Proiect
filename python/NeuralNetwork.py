import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.num_weights = (input_size * hidden_size) + hidden_size + \
                           (hidden_size * output_size) + output_size

    def forward(self, inputs, all_weights):
        w1_end = self.input_size * self.hidden_size
        w1 = all_weights[0: w1_end].reshape(self.input_size, self.hidden_size)

        b1_end = w1_end + self.hidden_size
        b1 = all_weights[w1_end: b1_end]

        w2_end = b1_end + (self.hidden_size * self.output_size)
        w2 = all_weights[b1_end: w2_end].reshape(self.hidden_size, self.output_size)

        b2 = all_weights[w2_end:]

        hidden_input = np.dot(inputs, w1) + b1
        hidden_output = self.sigmoid(hidden_input)

        final_input = np.dot(hidden_output, w2) + b2
        final_output = self.sigmoid(final_input)

        return final_output * 10

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))