"""Module for pytorch models."""

import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
        AlexNet is a convolutional neural network architecture developed for image classification tasks, 
        notably achieving prominence through its performance in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC).
        It classifies images into 1,000 distinct object categories and is regarded as the first widely recognized application
        of deep convolutional networks in large-scale visual recognition. 

        Source:
        https://en.wikipedia.org/wiki/AlexNet
    """
    def __init__(self, num_classes : int = 10):
        """
        Definition of the architecture.

        Args:
            - num_classess: the number of output classes
        """
        super().__init__()
        self.num_classes = num_classes
        # Architecture of the convolutional Layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # Architecture of the fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        """
        Input output mapping by the model.

        Args:
           -x (torch.Tensor): Input tensor that has to be classified

        Returns:
            torch.Tensor: tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits