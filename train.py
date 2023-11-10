import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type=str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args

def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data

def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data

def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader

def check_gpu(gpu_arg):
    if not gpu_arg:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device

def primaryloader_model(architecture="vgg16"):
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    
    for param in model.parameters():
        param.requires_grad = False 
    return model

def initial_classifier(model, hidden_units):
    classifier = nn.Sequential(OrderedDict([
                ('inputs', nn.Linear(25088, 120)),
                ('relu1', nn.ReLU()),
                ('dropout', nn.Dropout(0.5)),
                ('hidden_layer1', nn.Linear(120, 90)),
                ('relu2', nn.ReLU()),
                ('hidden_layer2', nn.Linear(90, 70)),
                ('relu3', nn.ReLU()),
                ('hidden_layer3', nn.Linear(70, 102)),
                ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    return classifier

def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

def network_trainer(Model, Trainloader, Testloader, Device, 
                  Criterion, Optimizer, Epochs, Print_every, Steps):
    
    if type(Epochs) == type(None):
        Epochs = 12
        print("Number of Epochs specificed as 12.")    
 
    print("Training process initializing .....\n")

    # Train Model
    for e in range(Epochs):
        running_loss = 0
        Model.train() 
        
        for ii, (inputs, labels) in enumerate(Trainloader):
            Steps += 1
            
            inputs, labels = inputs.to(Device), labels to(Device)
            
            Optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)
            
                print("Epoch: {}/{} | ".format(e+1, epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(testloader)),
                     "Validation Accuracy: {:.4f}".format(accuracy/len(testloader))
                
                running_loss = 0
                model.train()

    return Model

def validate_model(Model, Testloader, Device):
    correct, total = 0, 0
    with torch.no_grad():
        model.eval()
        for data in train_loader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            predicted = torch.max(outputs.data, 1)
           
