import torch
from torch import nn

# Optimizing the Model Parameters
class ModelTraining:
    def __init__(self, model, device_name):
        self.model = model
        self.device_name = device_name
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def train(self, dataloader):
        size = len(dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device_name), y.to(self.device_name)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device_name), y.to(self.device_name)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    def save_model(self, path: str = "pytorch-tutorials/model.pth"):
        # Saving Models
        torch.save(self.model.state_dict(), path)
        print(f"Saved PyTorch Model State to {path}")

def train_model(model, train_dataloader, test_dataloader, device_name, epochs=15):
    mt_obj = ModelTraining(model, device_name)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        mt_obj.train(train_dataloader)
        mt_obj.test(test_dataloader)
    print("Done!")
    mt_obj.save_model()

