"""
Be sure you have minitorch installed in your Virtual Env.
>>> pip install -Ue .
"""

import minitorch
import time  # for tracking time

def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.tensor(r, requires_grad=True)

class Linear(minitorch.Module):
    def __init__(self, in_features, out_features):
        """
        Initializes the Linear layer with weights and biases.
        """
        super().__init__()
        self.weight = self.add_parameter("weight", minitorch.rand((in_features, out_features)))
        self.bias = self.add_parameter("bias", minitorch.rand((out_features,)))

    def forward(self, x):
        """
        Performs the linear transformation.
        """
        return x @ self.weight.value + self.bias.value

class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        """
        Initializes the neural network with three linear layers.
        """
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        """
        Performs a forward pass through the network.
        """
        x = self.layer1.forward(x).relu()
        x = self.layer2.forward(x).relu()
        x = self.layer3.forward(x)
        return x.sigmoid()

def default_log_fn(epoch, total_loss, correct, total_samples, epoch_time):
    epoch_time_ms = epoch_time * 1000  # Convert to milliseconds
    print(f"Epoch {epoch} | Loss: {total_loss:.4f} | Correct: {correct}/{total_samples} | Epoch Time: {epoch_time_ms:.2f}ms")

class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(self.hidden_layers)

    def run_one(self, x):
        # Single prediction run
        input_tensor = minitorch.tensor([x], requires_grad=False)
        return self.model.forward(input_tensor)[0].item()  # Get the scalar value

    def run_many(self, X):
        # Batch prediction run
        input_tensor = minitorch.tensor(X, requires_grad=False)
        return self.model.forward(input_tensor)

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        # Convert data to tensors
        X = minitorch.tensor(data.X, requires_grad=False)
        y = minitorch.tensor(data.y, requires_grad=False)

        losses = []
        total_samples = len(data.X)
        total_training_time = 0.0  # Initialize total training time
        for epoch in range(1, self.max_epochs + 1):
            start_time = time.time()  # Start the timer for the epoch

            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward pass
            out = self.model.forward(X).view(data.N)  # Reshape to (N,)
            one_tensor = minitorch.tensor([1.0], requires_grad=False)
            prob = (out * y) + ((one_tensor - out) * (one_tensor - y))  # Ensure '1' is a tensor
            loss = -prob.log().mean()

            # Backward pass
            loss.backward()
            total_loss = loss.item()  # Get the scalar value of the loss
            losses.append(total_loss)

            # Update parameters
            optim.step()

            # Calculate correct predictions
            out_np = out.to_numpy()  # Convert tensor to NumPy array
            predictions = minitorch.tensor([float(v > 0.5) for v in out_np])  # Manually cast boolean to float
            correct = (predictions == y).sum().item()  # Get the sum of correct predictions as a scalar

            # End the timer for the epoch and calculate the elapsed time
            epoch_time = time.time() - start_time
            total_training_time += epoch_time  # Accumulate total training time

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, total_loss, correct, total_samples, epoch_time)

        # Print total training time in seconds
        print(f"Total training time: {total_training_time:.4f}s")

# Example usage
if __name__ == "__main__":
    PTS = 50
    HIDDEN = 10
    RATE = 0.5
    data = minitorch.datasets["Split"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
