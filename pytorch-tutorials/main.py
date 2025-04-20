import torch

from load_data import train_dataloader, test_dataloader, training_data, test_data
from nn_model import NeuralNetwork, device_name
from model_training import train_model


model = NeuralNetwork().to(device_name)
# train_model(model, train_dataloader, test_dataloader, device_name, 20)
model.load_state_dict(torch.load("pytorch-tutorials/model.pth"))

# 7. Making Predictions
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
for i in range(0, 20, 1):
    x, y = test_data[i][0], test_data[i][1]
    with torch.no_grad():
        x = x.to(device_name)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

