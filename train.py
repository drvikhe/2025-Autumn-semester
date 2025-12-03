import torch
import torchvision.datasets
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据准备
train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练集长度: {train_data_size}")
print(f"测试集长度: {test_data_size}")

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

# 模型定义
import torch
from torch import nn


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


tudui = Tudui()
tudui = tudui.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate, momentum=0.9)

# 记录参数
total_train_step = 0
total_test_step = 0
epoch = 30

writer = SummaryWriter("logs/train_loss")
start_time = time.time()

for i in range(epoch):
    print(f"第 {i + 1} 轮训练开始")
    tudui.train()

    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(f"耗时: {end_time - start_time}")
            print(f"训练次数: {total_train_step}, Loss: {loss.item()}")
            writer.add_scalar('train-loss', loss.item(), total_train_step)

    tudui.eval()

    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)

            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print(f"整体验证集 Loss: {total_test_loss}")
    print(f"整体验证集 Accuracy: {total_accuracy / test_data_size}")

    writer.add_scalar('test-loss', total_test_loss, total_test_step)
    writer.add_scalar('test-accuracy', total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # 保存模型
    torch.save(tudui, f"tudui_{i}.pth")
    print("模型已保存")

writer.close()
