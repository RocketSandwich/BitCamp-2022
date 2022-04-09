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
            nn.Linear(6, 320),
            nn.ReLU(),
            nn.Linear(320, 384),
            nn.ReLU(),
            nn.Linear(384, 352),
            nn.ReLU(),
            nn.Linear(352, 448),
            nn.ReLU(),
            nn.Linear(448, 160),
            nn.ReLU(),
            nn.Linear(160, 160),
            nn.ReLU(),
            nn.Linear(160, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, input):
        out = self.linear_relu_stack(input)
        return out

