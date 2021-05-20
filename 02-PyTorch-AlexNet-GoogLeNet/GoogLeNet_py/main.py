import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import time
import os
import copy
import torch.nn.functional as F
#%%
# Preprocess inputs to 3x32x32 with CIFAR-specific normalization parameters

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #### NOTE FOR PRETRAIN

# Download CIFAR-10 and set up train, validation, and test datasets with new preprocess object

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=preprocess)

train_datset, val_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=preprocess)

# Create DataLoaders

batch_size = 4

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

#%%
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, weights_name='weight_save', is_inception=False):
    '''
    train_model function

    Train a PyTorch model for a given number of epochs.
    
            Parameters:
                    model: Pytorch model
                    dataloaders: dataset
                    criterion: loss function
                    optimizer: update weights function
                    num_epochs: number of epochs
                    weights_name: file name to save weights
                    is_inception: The model is inception net (Google LeNet) or not

            Returns:
                    model: Best model from evaluation result
                    val_acc_history: evaluation accuracy history
                    loss_acc_history: loss value history
    '''
    since = time.time()

    val_acc_history = []
    loss_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()

        #if (epoch+1) % 5 == 0:
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over the train/validation dataset according to which phase we're in
            
            for inputs, labels in dataloaders[phase]:

                # Inputs is one batch of input images, and labels is a corresponding vector of integers
                # labeling each image in the batch. First, we move these tensors to our target device.
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero out any parameter gradients that have previously been calculated. Parameter
                # gradients accumulate over as many backward() passes as we let them, so they need
                # to be zeroed out after each optimizer step.

                optimizer.zero_grad()

                # Instruct PyTorch to track gradients only if this is the training phase, then run the
                # forward propagation and optionally the backward propagation step for this iteration.

                with torch.set_grad_enabled(phase == 'train'):
                    # The inception model is a special case during training because it has an auxiliary
                    # output used to encourage discriminative representations in the deeper feature maps.
                    # We need to calculate loss for both outputs. Otherwise, we have a single output to
                    # calculate the loss on.
                    
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_a4, aux_d4 = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_a4, labels)
                        loss3 = criterion(aux_d4, labels)
                        loss = loss1 + 0.3*(loss2+loss3)
#                     elif is_inception and weights_name == 'GoogLeNet' and phase == 'val':
#                         outputs, _, _ = model(inputs) #,_,_ or in the forward()
#                         loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs) #,_,_ or in the forward()
                        loss = criterion(outputs, labels)            
                    
                    outputs = nn.functional.softmax(outputs, dim=1)        
                    _, preds = torch.max(outputs, 1)

                    # Backpropagate only if in training phase

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Gather our summary statistics
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_end = time.time()
            
            elapsed_epoch = epoch_end - epoch_start
            
            #if (epoch+1) % 5 == 0:
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print("Epoch time taken: ", elapsed_epoch)
    
            # If this is the best model on the validation set so far, deep copy it

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), weights_name + ".pth")
            if phase == 'val':
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                loss_acc_history.append(epoch_loss)

#         print()

    # Output summary statistics, load the best weight set, and return results
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, loss_acc_history

#%%
def evaluate(model, iterator, criterion,model_name):
    
    total = 0
    correct = 0
    epoch_loss = 0
    epoch_acc = 0
    
    predicteds = []
    trues = []
    
    model.eval()
    
    with torch.no_grad():
    
        for batch, labels in iterator:
            
            #Move tensors to the configured device
            batch = batch.to(device)
            labels = labels.to(device)

#             if model_name == "GoogLeNet":
#                 predictions, _, _ = model(batch.float())
#             else:
            predictions = model(batch.float())
                
            loss = criterion(predictions, labels.long())
            
            predictions = nn.functional.softmax(predictions, dim=1)            
            _, predicted = torch.max(predictions.data, 1)  #returns max value, indices
                       
            predicteds.append(predicted)
            trues.append(labels)            
            total += labels.size(0)  #keep track of total
            correct += (predicted == labels).sum().item()  #.item() give the raw number
            acc = 100 * (correct / total)
            
            epoch_loss += loss.item()
            epoch_acc += acc
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator),predicteds, trues

#%%
from Modules import GoogLeNet

googlenet = GoogLeNet()
googlenet_pre = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True, aux_logits = True)
models = [googlenet, googlenet_pre]
model_names = ['GoogLeNet', 'Pretrained GoogLeNet']
# models = [googlenet_pre]
# model_names = ['Pretrained GoogLeNet']

#%%
criterion = nn.CrossEntropyLoss()

optimizers = []
for model in models:
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizers.append(optimizer)

#%%
from chosen_gpu import get_freer_gpu
device = torch.device(get_freer_gpu()) if torch.cuda.is_available() else torch.device("cpu")
print("Configured device: ", device)

#%%

for model in models:
    model = model.to(device)
    
criterion = criterion.to(device)

#%%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

for i,model in enumerate(models):
    print(f'The model {model_names[i]} has {count_parameters(model):,} trainable parameters')# Train the model

#%%
dataloaders = { 'train': train_dataloader, 'val': val_dataloader }

#%%

print(f'Training: {model_names[0]}')
best_model, val_acc_history, loss_acc_history = train_model(models[0], dataloaders, criterion, optimizers[0], 1, model_names[0],is_inception=True )
#plot_data(val_acc_history, loss_acc_history)

#%%
print(f'Training: {model_names[1]}')
best_model, val_acc_history, loss_acc_history = train_model(models[1], dataloaders, criterion, optimizers[1], 1, model_names[1],is_inception=True )
#plot_data(val_acc_history, loss_acc_history)

#%%
for i, model in enumerate(models):
    print(f'Model: {model_names[i]}')
    model.load_state_dict(torch.load(f'{model_names[i]}.pth'))
    test_loss, test_acc, test_pred_label, test_true_label  = evaluate(model, test_dataloader, criterion, model_names[i])
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')

