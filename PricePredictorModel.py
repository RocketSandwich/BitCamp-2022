import torch.nn as nn
import numpy as np

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

class PricePredictor(nn.Module):

    def __init__(self):
        super(PricePredictor, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(29, 418),
            nn.ReLU(),
            nn.Linear(418, 380),
            nn.ReLU(),
            nn.Linear(380, 380),
            nn.ReLU(),
            nn.Linear(380, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, input):
        out = self.linear_relu_stack(input)
        return out

