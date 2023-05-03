import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import models
import torchvision
import os
import sys
from tqdm import tqdm
import math
import numpy as np
from torchsummary import summary
from datetime import datetime
import csv

base_dir = os.getcwd()
device = torch.device('cuda')

frozen = False

# These are the variables that will be changed
# In order to stop the training early, do CTRL+C
learning_rate = 1e-5 #1e-5
batch_size = 10 # 10
num_epochs = 100 # 100 / 300
Trained_Model = 2  # ResNet50, VGG16, InceptionV3 (1-3)
Train_type = 2 # Train whole Model with no Weights, Train w/ Freeze, Train w/ Freeze then Unfreeze  (1-3)

ContinueTrain = False

total_train_hr = 0
total_train_min = 0
total_train_sec = 0

# Defining the model, their corresponding transformations, and adding layers to transform the model to 100 classes
# Each transformation parameter was taken from their respective documentation page https://pytorch.org/vision/main/models.html 
if (Trained_Model == 1):
    if Train_type != 1:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=None)
    model.to(device)
    # Transformations applied to the training set are often randomized to reduce over-fitting
    transform = transforms.Compose([
        transforms.Resize(232),
        transforms.RandomCrop(224, pad_if_needed=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=(0.5, 1.5), contrast=1, saturation=(0.5, 1.5), hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Transformations applied to the test set aren't changed, this is just moving it to a format the model can understand
    transform_test = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Adding onto the model
    Classifier = nn.Sequential(
        nn.Linear(2048, 1000),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1000, 256), 
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(256, 100))
    
    Direc = os.path.join(base_dir, 'ResNet50')
elif (Trained_Model == 2):
    if Train_type != 1:
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    else:
        model = models.vgg16(weights=None)
    model.to(device)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224, pad_if_needed=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=(0.5, 1.5), contrast=1, saturation=(0.5, 1.5), hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.48235, 0.45882, 0.40784], [0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.48235, 0.45882, 0.40784], [0.00392156862745098, 0.00392156862745098, 0.00392156862745098])
    ])
    Classifier = nn.Sequential( #model.classifier[6]
        nn.Linear(4096, 256),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(256, 100)
    )
    
    Direc = os.path.join(base_dir, 'VGG16')
elif (Trained_Model == 3):
    if Train_type != 1:
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    else:
        model = models.inception_v3(weights=None)
    model.to(device)
    transform = transforms.Compose([
        transforms.Resize(342),
        transforms.RandomCrop(299, pad_if_needed=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=(0.5, 1.5), contrast=1, saturation=(0.5, 1.5), hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    Classifier = nn.Sequential(
        nn.Linear(2048, 1000),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1000, 256), 
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(256, 100))
        
    Direc = os.path.join(base_dir, 'InceptionV3')
    
    model.aux_logits = False
    
# Moving the network model's processing to the GPU and setting it to train mode
model.to(device)
model.train()
    
# Downloading and loading the CIFAR100 dataset from : https://www.cs.toronto.edu/~kriz/cifar.html (python version)
DLTraining = datasets.CIFAR100(root=os.path.join(base_dir, 'CIFAR100'), train=True, download=True, transform=transform)
DLTest = datasets.CIFAR100(root=os.path.join(base_dir, 'CIFAR100'), train=False, download=True, transform=transform_test)

TrainingData = DataLoader(dataset=DLTraining, batch_size=batch_size, shuffle=True)
TestData = DataLoader(dataset=DLTest, batch_size=batch_size, shuffle=True)

avgtestloss = []
def get_accuracy(loader, model):
    correct_count = 0
    total = 0
    model.eval() # Set the model to evaluation mode
    testlosses = []
    
    bar = tqdm(loader, total=len(loader), leave=False)
    
    with torch.no_grad():
        for x, y in bar:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            if loader.dataset.train:
                bar.set_description("Getting Accuracy for: Training Set")
            else:
                loss = crit(scores, y)
                testlosses.append(loss.item())
                bar.set_postfix(loss= loss.item()) 
                bar.set_description("Getting Accuracy for Testing Set")
                
            _, predictions = scores.max(1)
            correct_count += (predictions == y).sum()
            total += predictions.size(0)
    model.train()
    if loader.dataset.train == False:
        avgloss = sum(testlosses)/len(testlosses)
        avgtestloss.append(avgloss)
        print(f"Avg Testing Loss: {avgloss:.2f}")
    return (correct_count/total) # Get back total accuracy on the data



def checkpointcreate(state, epoch):
    str1 = 'checkpoint_'
    if (frozen == True):
        str2 = '_frozen.tar'
    else:
        str2 = '_.Unfrozen.tar'
    checkpoint_string = str1 + str(epoch) + str2
    checkpoint_filename = os.path.join(Direc, checkpoint_string)
    
    print("Checkpoint Saved.")
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()},
        checkpoint_filename
    )

    
# All of these are appended after each epoch
loss_over_time = []
accuracy_test_over_time = []
accuracy_trainset_over_time = []
time_taken = []

start_train = datetime.now()
Epoch_save = 0

# Training the model
def Train(num_epoch):
    total_train_hr = 0
    total_train_min = 0
    total_train_sec = 0
    try:
        model.to(device)
        for epoch in range(num_epoch):
            model.train()
            start_epoch = datetime.now()
            losses = []
            
            bar = tqdm(enumerate(TrainingData), total=len(TrainingData), leave=False)
            bar.set_description(f"Epoch {epoch + 1}/{num_epoch}")
            num_correct = 0
            total_data = 0
            
            for batch_idx, (data, label) in bar:
                data = data.to(device)
                label = label.to(device)
                
                optimizer.zero_grad()
                
                score = model(data)
                _, prediction = score.max(1)
                
                loss = crit(score, label)
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
                num_correct += (prediction==label).sum()
                total_data += prediction.size(0)
                
                bar.set_postfix(loss= loss.item())
                
            # Get time difference in between each epoch
            CurrentTime = datetime.now()
            diff = CurrentTime - start_epoch
            diff = diff.total_seconds()
            
            hours = math.floor(diff / 60**2) # Get amount of hours
            if hours > 0:
                diff -= (hours * (60**2))
            minutes = math.floor(diff / 60) # Get amount of minutes
            if minutes > 0:
                diff -= (minutes * 60)
                
            print(f"Time To Train: {hours} Hours, {minutes} Minutes, {diff:.2f} Seconds | Epoch {epoch+1}")
            time = (f"{hours} | {minutes} | {diff:.2f}")
            time_taken.append(time)
            
            total_train_hr += (hours)
            total_train_min += (minutes)
            total_train_min += (diff)
                
            acctrain = get_accuracy(TrainingData, model)*100
            acctest = get_accuracy(TestData, model)*100
            avgloss = sum(losses)/len(losses)
            print(f"Epoch {epoch + 1}/{num_epoch}: Accuracy [Used, Training, Test]: {num_correct/total_data*100:.2f}% / {acctrain:.2f}% / {acctest:.2f}% | Loss Avg: {avgloss:.3f}")
            
            # Get time difference in between each epoch
            CurrentTime = datetime.now()
            diff = CurrentTime - start_epoch
            diff = diff.total_seconds()
            
            hours = math.floor(diff / 60**2) # Get amount of hours
            if hours > 0:
                diff -= (hours * (60**2))
            minutes = math.floor(diff / 60) # Get amount of minutes
            if minutes > 0:
                diff -= (minutes * 60)
                
            print(f"Time After Accuracy: {hours} Hours, {minutes} Minutes, {diff:.2f} Seconds | Epoch {epoch+1}")
            
            # Processing Data to be Graphed and put into a Spreadsheet
            acctrain = acctrain.numpy(force=True)
            acctrain = acctrain.astype(np.float16)
            accuracy_trainset_over_time.append(acctrain)
            
            acctest = acctest.numpy(force=True)
            acctest = acctest.astype(np.float16)
            accuracy_test_over_time.append(acctest)
            
            loss_over_time.append(avgloss)
            
            # Saving the current model after each epoch (checkpoint)
            checkpoint_save = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
            checkpointcreate(checkpoint_save, epoch=(epoch+1))
            print("")
    except KeyboardInterrupt:
        print("Early stop!")
        pass
    

if Train_type == 1: # Before training, declare CrossEntropyLoss and Adam along with any learning rate at the time
    crit = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if Trained_Model == 2:
        model.classifier[6] = nn.Linear(4096, 100)
    else:
        model.fc = nn.Linear(2048, 100)
        
    if ContinueTrain:
        filename = os.path.join(base_dir, 'continue.tar')
        print("File found!")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    Train(num_epochs)
    
elif Train_type == 2:
    if Trained_Model == 1:
        for param in model.parameters():
            param.requires_grad = False
                
        model.fc = Classifier
        
    if Trained_Model == 2:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = Classifier
        
    elif Trained_Model == 3:
        for param in model.parameters():
            param.requires_grad = False
        model.fc = Classifier
        
    print("Model Frozen, Training Fewer Layers | Training Method 2")
    crit = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    Train(num_epochs)
    
elif Train_type == 3:
    if Trained_Model == 2:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = Classifier
    else:
        for param in model.parameters():
            param.requires_grad = False
        model.fc = Classifier
        
    print("Model Frozen, Training Fewer Layers")
    # Freeze everything except classifier and train with lower learning rate
    
    crit = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate/10)

    epochs = (num_epochs * (1/4))
    frozen = True
    Train(num_epoch=int(epochs))
    frozen = False
    # Unfreeze the whole model and retrain with a higher training rate
    for param in model.parameters():
        param.requires_grad = True
            
    print("Model Unfrozen, Adjusted Learning Rate.")

    crit = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    epochs = (num_epochs * (3/4))
    Train(num_epoch=int(epochs))
    
# After Training
CurrentTime = datetime.now()
diff = CurrentTime - start_train
diff = diff.total_seconds()

hours = math.floor(diff / 60**2) # Get amount of hours
if hours > 0:
    diff -= (hours * (60**2))
minutes = math.floor(diff / 60) # Get minutes
if minutes > 0:
    diff -= (minutes * 60)
    
print("")
print("Training Completed!")        
print(f"Time for Whole Training: {hours} Hours, {minutes} Minutes, {diff:.2f} Seconds")

print("")
print("Final Accuracy:")
print(f"Training Set: {get_accuracy(TrainingData, model)*100:.2f}% | Testing Set: {get_accuracy(TestData, model)*100:.2f}%")

# Append Everything to a CSV File
Final_Test_Acc = get_accuracy(TestData, model)*100
Final_Test_Acc.numpy(force=True).astype(np.float16)

Final_Train_Acc = get_accuracy(TrainingData, model)*100
Final_Train_Acc.numpy(force=True).astype(np.float16)

Epoch = []
for x in range(num_epochs):
    Epoch.append(x+1)
    
CSVDirec = os.path.join(Direc, 'data.csv')
fields = ["Epoch", "Loss", "Training Accuracy", "Testing Accuracy", "Testing Loss", "Time Taken"]
with open(CSVDirec, 'w', newline='') as new_file:
    csv_writer = csv.DictWriter(new_file, fieldnames=fields, delimiter=",")
    csv_writer.writeheader()
    for x in range(len(loss_over_time)):
        csv_writer.writerow({"Epoch": Epoch[x],
                             "Loss": loss_over_time[x],
                             "Training Accuracy": accuracy_trainset_over_time[x],
                             "Testing Accuracy": accuracy_test_over_time[x],
                             "Testing Loss": avgtestloss[x],
                             "Time Taken": time_taken[x]})
    csv_writer.writerow({"Epoch": "Final Averages",
                         "Loss": "",
                         "Training Accuracy": Final_Train_Acc,
                         "Testing Accuracy": Final_Test_Acc,
                         "Time Taken": f"Total Time: {hours} Hours, {minutes} Minutes, {diff:.2f} Seconds"})
    csv_writer.writerow({"Epoch": "",
                         "Loss": "",
                         "Training Accuracy": "",
                         "Testing Accuracy": "",
                         "Time Taken": f"Total Training Time: {total_train_hr} Hours, {total_train_min} Minutes, {total_train_sec:.2f} Seconds"})

print("CSV Completed.")

best_acc = max(accuracy_test_over_time)
best_epoch = accuracy_test_over_time.index(best_acc)+1

print(f"Best Accuracy on Test Set was {best_acc}% at epoch {best_epoch}")
