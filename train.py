# Trains the model. does training and validation
# Saves the best version of the model with the lowest loss in validation

from torchvision import datasets, transforms
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim
from model import build_model

MODEL_PATH = "mnist_mlp.pt"

def evaluate(model, loader, name):
    model.eval()                          #set to eval mode
    total_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            #Track metrics (extract values using .item() to prevent memory leaks)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
     
        #Calculate final metrics
        avg_loss = total_loss / len(loader.dataset)
        accuracy = 100.0 * correct / total
        print(f"{name} Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        return avg_loss

train_data = datasets.MNIST(root= "data" , train=True, download= True, transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=0.1307,std=0.3081)]))
test_data = datasets.MNIST(root= "data" , train=False, download= True, transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=0.1307,std=0.3081)]))

generator = torch.Generator().manual_seed(42)
train_len = 54000
val_len = 6000
train_set , val_set = random_split(train_data, [train_len,val_len], generator=generator)
torch.manual_seed(42)
model = build_model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_Loader = DataLoader(dataset=train_set,batch_size=64,shuffle=True,drop_last=True)
val_Loader = DataLoader(dataset=val_set,batch_size=128)
test_Loader = DataLoader(dataset=test_data,batch_size=128)

# Train and Validation loop - goes for 20 epochs. saves the best version to prevent over/under fitting
lowest_val_loss = float("inf")                # impossibly high number - will immediately get beaten
for i in range(20):
    print("Current epoch: ",i)
    tot_loss = 0
    to_div = 0 
    for inputs, targets in train_Loader:
        optimizer.zero_grad(set_to_none=True)       #resets the gradients. set_to_none for memory
        outputs = model(inputs)
        loss = criterion(outputs,targets)
        tot_loss += loss.item()
        to_div += 1
        loss.backward()                    #calculates gradients and adds them to .grad buffers
        optimizer.step()                    #updates the current weights
    print("avg train loss: ", tot_loss/to_div)
    avg_val_loss = evaluate(model,val_Loader,"Validation")

    if (avg_val_loss < lowest_val_loss):
        lowest_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print("Saved current best model")
    model.train()                         #return to train mode

# Testing phase - tests the best weights found only one time

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
test_loss = evaluate(model,test_Loader, "Test")



