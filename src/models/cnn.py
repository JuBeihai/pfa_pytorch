import torch
import torch.nn as nn

class MNISTCNN(nn.Module):
    """
    完全匹配 TensorFlow 版本的 CNN 架构
    对应 __cnn_mnist 函数
    """
    def __init__(self, num_classes=10):
        super().__init__()
        # Conv2D(16, 8, strides=2, padding='same', activation='relu')
        self.conv1 = nn.Conv2d(1, 16, 8, stride=2, padding=3)  # padding=3 for 'same'
        # MaxPool2D(2, 1)
        self.pool1 = nn.MaxPool2d(2, 1)
        # Conv2D(32, 4, strides=2, padding='valid', activation='relu')
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=0)  # padding=0 for 'valid'
        # MaxPool2D(2, 1)
        self.pool2 = nn.MaxPool2d(2, 1)
        # Flatten()
        self.flatten = nn.Flatten()
        # Dense(32, activation='relu')
        # 计算输入维度: 32 * 4 * 4 = 512
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        # Dense(10)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 输入形状: (batch_size, 784) -> (batch_size, 1, 28, 28)
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Conv2D(16, 8, strides=2, padding='same', activation='relu')
        x = self.relu(self.conv1(x))  # (batch, 1, 28, 28) -> (batch, 16, 14, 14)
        # MaxPool2D(2, 1)
        x = self.pool1(x)  # (batch, 16, 14, 14) -> (batch, 16, 13, 13)
        # Conv2D(32, 4, strides=2, padding='valid', activation='relu')
        x = self.relu(self.conv2(x))  # (batch, 16, 13, 13) -> (batch, 32, 5, 5)
        # MaxPool2D(2, 1)
        x = self.pool2(x)  # (batch, 32, 5, 5) -> (batch, 32, 4, 4)
        # Flatten()
        x = self.flatten(x)  # (batch, 32, 4, 4) -> (batch, 512)
        # Dense(32, activation='relu')
        x = self.relu(self.fc1(x))  # (batch, 512) -> (batch, 32)
        # Dense(10)
        x = self.fc2(x)  # (batch, 32) -> (batch, 10)
        return x