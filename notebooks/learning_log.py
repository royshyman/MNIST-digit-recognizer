# Learning log: the script I wrote step by step while learning to train my first neural network.
# Earlier steps are kept in ''' blocks. The cleaned-up version is train.py in the project root.
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch
from torch.utils.data import random_split, DataLoader
import torch.nn as nn
import torch.optim as optim

'''
train_data = datasets.MNIST(root= "data" , train=True, download= True, transform= transforms.ToTensor())

print(len(train_data))
img, label = train_data[0]
print(label)
print(img.shape)
print(img.dtype)
print(img.min())
print(img.max())

fig, axes = plt.subplots(2, 3)   # 2 rows, 3 columns → axes is a 2D array of 6 plots
for i, ax in enumerate(axes.flat):   # .flat lets you loop over all 6 as one list
    img, label = train_data[i]
    ax.imshow(img.squeeze(), cmap="gray")
    ax.set_title(label)
    ax.axis("off")

plt.show()
'''

'''
train_len = 54000
val_len = 6000
generator = torch.Generator().manual_seed(42)
train_set , val_set = random_split(train_data, [train_len,val_len], generator=generator)
print(len(train_set), len(val_set))
img_list = []
for i in range(len(train_set)):
    img , label = train_data[i]
    img_list.append(img)
stacked = torch.stack(img_list)
print(stacked.mean())
print(stacked.std())

'''
train_data = datasets.MNIST(root= "data" , train=True, download= True, transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=0.1307,std=0.3081)]))
generator = torch.Generator().manual_seed(42)
train_len = 54000
val_len = 6000
train_set , val_set = random_split(train_data, [train_len,val_len], generator=generator)
'''


img_list = []
for i in range(len(train_set)):
    img , label = train_set[i]
    img_list.append(img)
stacked = torch.stack(img_list)
print(stacked.mean())
print(stacked.std())
'''

model = nn.Sequential(nn.Flatten(),nn.Linear(784,128),nn.ReLU(),nn.Linear(128,10))
print(model[1].weight.shape)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1.0)
train_Loader = DataLoader(dataset=train_set,batch_size=64,shuffle=True,drop_last=True)
'''
data_iter = iter(train_Loader)
imgs, labels = next(data_iter)
outputs = model(imgs)
print(imgs.shape)
print(labels.shape)
print(outputs.shape)
loss = criterion(outputs,labels)
print(loss)
'''
train_loss = []
val_loss = []
curr_epoch = 0
for i in range(10):
    curr_epoch += 1
    print("Current epoch: ",curr_epoch)
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
    print("avg loss: ", tot_loss/to_div)
    train_loss.append(tot_loss/to_div)
    val_Loader = DataLoader(dataset=val_set,batch_size=128)
    model.eval()                          #set to eval mode
    total_val_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in val_Loader:
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            #Track metrics (extract values using .item() to prevent memory leaks)
            total_val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    #Calculate final metrics
    avg_val_loss = total_val_loss / len(val_Loader.dataset)
    val_accuracy = 100.0 * correct / total
    print(f"Validation Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.2f}%")
    val_loss.append(avg_val_loss)
    model.train()                         #return to train mode

epochs = range(1,len(train_loss)+1)
plt.plot(epochs,train_loss, label= "Train")
plt.plot(epochs,val_loss, label= "Validation")
plt.xlabel("Epochs")
plt.ylabel("Average loss")
plt.legend()
plt.show()

'''
test_data = datasets.MNIST(root= "test" , train=False, download= True, transform= transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=0.1307,std=0.3081)]))
test_Loader = DataLoader(dataset=test_data,batch_size=128)
total_test_loss = 0
total = 0
correct = 0
wrong_imgs, wrong_true, wrong_pred = [], [], []
model.eval()
with torch.no_grad():
        for inputs, targets in test_Loader:
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            #Track metrics (extract values using .item() to prevent memory leaks)
            total_test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            mask = predicted != targets            # True where the model was wrong
            wrong_imgs.append(inputs[mask])
            wrong_true.append(targets[mask])
            wrong_pred.append(predicted[mask])
#Calculate final metrics
avg_test_loss = total_test_loss / len(test_Loader.dataset)
test_accuracy = 100.0 * correct / total
print(f"Test Loss: {avg_test_loss:.4f} | Accuracy: {test_accuracy:.2f}%")

wrong_imgs = torch.cat(wrong_imgs)
wrong_true = torch.cat(wrong_true)
wrong_pred = torch.cat(wrong_pred)
print("mistakes:", len(wrong_imgs))

fig, axes = plt.subplots(3, 4)
for i, ax in enumerate(axes.flat):
    ax.imshow(wrong_imgs[i].squeeze(), cmap="gray")
    ax.set_title(f"true {wrong_true[i].item()} / pred {wrong_pred[i].item()}")
    ax.axis("off")
plt.tight_layout()
plt.show()
'''



