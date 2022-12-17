import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import process_data as processor
from sklearn.metrics import confusion_matrix
import seaborn as sns

processor.go()

X_train = torch.from_numpy((processor.X_train_sklearn).astype(float)).type(torch.FloatTensor)
y_train = torch.from_numpy(processor.y_train_sklearn.astype(int)).type(torch.LongTensor)
X_test = torch.from_numpy(processor.X_test_sklearn.astype(float)).type(torch.FloatTensor)
y_test = torch.from_numpy(processor.y_test_sklearn.astype(int)).type(torch.LongTensor)

ds = TensorDataset(X_train, y_train)
ts = TensorDataset(X_test, y_test)
batch_size = 64

train_dataLoader = DataLoader(ds, batch_size=batch_size)
test_dataLoader = DataLoader(ts, batch_size=batch_size)

class BBallNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(13, 13),
            nn.Identity(),
            nn.Linear(13, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Sigmoid(),
            nn.Linear(256, 2)

        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = BBallNeuralNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataLoader, model, loss_fn, optimizer)
    test(test_dataLoader, model, loss_fn)
    if t == 9:
        y_pred = model(X_test).argmax(1)
        c_matrix = confusion_matrix(y_test, y_pred)

        sns.heatmap(c_matrix, annot=True, fmt='d')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.show()
    
print("Done!")