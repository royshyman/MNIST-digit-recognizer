# Builds the model used
import torch.nn as nn

def build_model():
    model = nn.Sequential(nn.Flatten(),nn.Linear(784,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,10))
    return model