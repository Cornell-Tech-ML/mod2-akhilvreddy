"""
Be sure you have minitorch installed in your Virtual Env.
>>> pip install -Ue .
"""

import minitorch
import random
import time
import numpy as np
from typing import Any

from minitorch.tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)
from minitorch.tensor_data import TensorData, Shape, Strides, Storage
from minitorch.tensor_ops import SimpleBackend, TensorBackend

# Use this function to make a random parameter in your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        h = self.layer1.forward(x).relu()
        h = self.layer2.forward(h).relu()
        return self.layer3.forward(h).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        reshaped_x = x.view(*x.shape, 1)
        reshaped_weights = self.weights.value.view(
            1, *self.weights.value.shape
        )
        multiplied = reshaped_x * reshaped_weights
        summed = multiplied.sum(dim=1).contiguous()
        summed = summed.view(x.shape[0], self.out_size)
        bias_reshaped = self.bias.value.view(1, self.out_size)
        out = summed + bias_reshaped

        return out

def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)

class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        overall_start_time = time.time()
        for epoch in range(1, self.max_epochs + 1):
            epoch_start_time = time.time()

            total_loss = 0.0
            correct = 0
            optim.zero_grad()


            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)


            optim.step()

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)
                print(f"Epoch {epoch} took {epoch_duration:.4f} seconds")

        overall_end_time = time.time()
        total_time = overall_end_time - overall_start_time
        print(f"Training completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)