import torch
import random as rand
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.nn.functional import mse_loss
import pickle as pkl
import PricePredictorModel as modelNet
import matplotlib.pyplot as plt

with open("data.pkl", "rb") as File:
    xData, yData = pkl.load(File)

xData = torch.tensor(xData).float()
yData = torch.tensor(yData/100_000).float()

X_train, X_test, y_train, y_test = train_test_split(xData, yData, test_size=.15, random_state=1002)

try:
    checkpoint = torch.load("PricePredictor.mdl")
except:
    checkpoint = None
    
#Building the network
learningRate = 3e-2
batch_size = 128
num_epochs = 50
testSize = 0.15
seed = 10005

#network = nn.neuralNet()
network = modelNet.PricePredictor()

if checkpoint is not None:
    network.load_state_dict(checkpoint['model_state_dict'])

optim = torch.optim.Adam(network.parameters(), lr=learningRate)

if checkpoint is not None:
    optim.load_state_dict(checkpoint['optimizer_state_dict'])

trainError = []
testError = []
print("Starting to train")
for epoch in range(num_epochs):
    for X_batch,y_batch in modelNet.iterate_minibatches(X_train, y_train, batchsize=batch_size, shuffle=True):
        network.zero_grad()
        out = network(X_batch.view(-1, 6))
        loss = mse_loss(out, y_batch.view(-1, 1))
        loss.backward()
        optim.step()

    network.eval()
    with torch.no_grad():
        error = []
        for X_batch,y_batch in modelNet.iterate_minibatches(X_train, y_train, batchsize=batch_size, shuffle=True):
            out = network(X_batch.view(-1, 6))
            loss = mse_loss(out, y_batch.view(-1, 1))
            error.append(loss.item())
        trainError.append(np.mean(error))

        error = []
        for X_batch,y_batch in modelNet.iterate_minibatches(X_test, y_test, batchsize=batch_size, shuffle=True):
            out = network(X_batch.view(-1, 6))
            loss = mse_loss(out, y_batch.view(-1, 1))
            error.append(loss.item())
        testError.append(np.mean(error))
    testInt = rand.randrange(0, len(y_test))
    print(f"Actual {int(y_test[testInt]*100000)} vs Predicted {int((network(X_test[testInt].view(-1, 6)).item())*100000)}")
    print(f"The training error is {trainError[-1]}\nThe testing error is {testError[-1]}")
    network.train()
    if epoch % 5 == 0:
        torch.save({
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            }, "PricePredictor.mdl")

torch.save({
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            }, "PricePredictor.mdl")

plt.plot(trainError, label='Train Error')
plt.plot(testError, label='Test Error')
plt.legend(loc='best')
plt.grid()
plt.show()