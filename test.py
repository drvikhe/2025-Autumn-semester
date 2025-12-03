import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Sequential
from torchvision import transforms

from nn_module import output

image_path = './image/dog.jpg'
image = Image.open(image_path)
print(image)

transform = transforms.Compose([torchvision.transforms.Resize((32,32)),
                                transforms.ToTensor()])
image = transform(image)
print(image.shape)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

model = torch.load("tudui_32.pth", weights_only=False,map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
probability = torch.nn.functional.softmax(output, dim=1)
print(probability)
prediction = output.argmax(1)

print(prediction)