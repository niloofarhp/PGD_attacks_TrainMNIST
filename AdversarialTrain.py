import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

def MNIST_DataLoader(batch_size_train = 128, batch_size_test = 6):
  train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/Users/niloofar/Documents/Projects/PGD_attacks_TrainMNIST/Dataset/', train=True, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                #torchvision.transforms.Resize((256, 256)),
                                #torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
                                #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
    batch_size=batch_size_train, shuffle=True)

  test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/Users/niloofar/Documents/Projects/PGD_attacks_TrainMNIST/Dataset/', train=False, download=True,
                              transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                #torchvision.transforms.Resize((256, 256)),
                                #torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
                                #torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])),
    batch_size=batch_size_test, shuffle=True)
  return train_loader, test_loader

def showSampledData(model, example_data):
  fig = plt.figure()
  example_targets = model(example_data.to(device))
  for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Pred word: {}".format(example_targets[i,:].argmax()))
    plt.xticks([])
    plt.yticks([])
  plt.show()
  return
class AdversarialPGD_Attack():
  def __init__(self):
    pass
  def PGD_attack(self,model, images, labels, eps=0.3, alpha=0.02, iters=20):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()        
    ori_images = images.data  
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()
        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    # showSampledData(model,images)                 
    return images

  def find_target(self, outputs,labels):
    predLabels = torch.zeros_like(outputs)
    for i in range(len(outputs)):
      predOuts = [0] * len(outputs[0])
      if outputs[i].argmax() == labels[i]:
        outputs[i,outputs[i].argmax()] = 0
        predLabels[i,outputs[i].argmax()] = 1
      else:
        predLabels[i,outputs[i].argmax()] = 1
    return predLabels

  def PGD_Targeted_Attack(self, model, images, labels, eps = 0.3, alpha = 0.02, iters=20,targLabels = 2):
    delta = torch.zeros_like(images, requires_grad=True)
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()        
    ori_images = images.data  
    for i in range(iters) :    
        images.requires_grad = True
        outputs = model(images)
        model.zero_grad()
        cost = (outputs.sum() - self.find_target(outputs,labels).sum()).to(device)
        cost.backward()
        adv_images = images + alpha*images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()    
    #showSampledData(model,images)     
    return images

  def No_AttackTrain(self,model, dataLoader):
    attackType = 'targeted'
    opt = optim.SGD(model.parameters(), lr=1e-2)
    epochs = 100
    acc = []
    lossR = []
    for epc in range(epochs):
      total_loss, total_err = 0.,0.
      for X,y in dataLoader:
          X,y = X.to(device), y.to(device)
          if(attackType == 'targeted'):
            X = self.PGD_Targeted_Attack(model, X,y)
          else:
            X = self.PGD_attack(model, X,y)
          yp = model(X)
          loss = nn.CrossEntropyLoss()(yp,y)
          opt.zero_grad()
          loss.backward()
          opt.step()          
          total_err += (yp.max(dim=1)[1] != y).sum().item()
          total_loss += loss.item() * X.shape[0]
      totalerror =  total_err / len(dataLoader.dataset), total_loss / len(dataLoader.dataset)  
      acc.append(1 - totalerror[0])
      lossR.append(totalerror[1])
      print("total accuracy : ",1 - totalerror[0], "total loss:",totalerror[1], sep="\t")
    fig, ax = plt.subplots(2,1)
    ax[0].plot(lossR, color='b', label="Training Loss")
    #ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)
    ax[1].plot(acc, color='b', label="Training Accuracy")
    #ax[1].plot(history.history['val_acc'], color='r',label="Validation Accuracy")
    legend = ax[1].legend(loc='best', shadow=True)

  def AttackEvaluate(self, model, dataLoader):
      eps = [0, 0.1, 0.2, 0.3, 0.45]
      alpha = 0.01
      pgdAttack = AdversarialPGD_Attack()
      attackType = 'nontargeted'
      opt = optim.SGD(model.parameters(), lr=1e-2)
      model.eval()
      acc = []
      lossR = []
      for epsilon in eps:
        total_loss, total_err = 0.,0.
        for X,y in dataLoader:
            X,y = X.to(device), y.to(device)
            if(attackType == 'targeted'):
                X = pgdAttack.PGD_Targeted_Attack(model, X,y, epsilon, alpha, 40)
            else:
                X = pgdAttack.PGD_attack(model, X,y, epsilon, alpha, 40)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp,y)       
            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]
        totalerror =  total_err / len(dataLoader.dataset), total_loss / len(dataLoader.dataset)  
        acc.append(1 - totalerror[0])
        lossR.append(totalerror[1])
      print("total accuracy : ",1 - totalerror[0], "total loss:",totalerror[1], sep="\t")
      fig, ax = plt.subplots(2,1)
      ax[0].plot(lossR, color='b', label="Training Loss")
      legend = ax[0].legend(loc='best', shadow=True)
      ax[1].plot(acc, color='b', label="Training Accuracy")
      legend = ax[1].legend(loc='best', shadow=True)
      return

class AdversarialTrain():
  def __init__(self):
    pass
  

model = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(7*7*64, 100), nn.ReLU(),
                        nn.Linear(100, 10)).to(device)
# process Dataset MNIST 
trainData, TestData = MNIST_DataLoader()
examples = enumerate(TestData)
batch_idx, (example_data, example_targets) = next(examples)
# model = torchvision.models.alexnet(pretrained=True).to(device)
# for param in model.features.parameters():
#   param.requires_grad = False
# for param in model.classifier.parameters():
#   param.requires_grad = False
# for param in model.classifier[6].parameters():
#   param.requires_grad = True
# model.classifier[6] = nn.Linear(4096, 10, bias = True)
print(model)
PGD_Attack = AdversarialPGD_Attack()
#showSampledData(model,example_data)
#attackedImgs = PGD_Attack.PGD_attack(model, example_data, example_targets)
#showSampledData(model,attackedImgs.cpu())
# attackedImgs = PGD_Attack.PGD_Targeted_Attack(model, example_data, example_targets)
# showSampledData(model,attackedImgs.cpu())

# PGD_Attack.No_AttackTrain(model,trainData)
# showSampledData(model,example_data)
# attackedImgs = PGD_Attack.PGD_attack(model, example_data, example_targets)
# showSampledData(model,attackedImgs.cpu())
# attackedImgs = PGD_Attack.PGD_Targeted_Attack(model, example_data, example_targets)
# showSampledData(model,attackedImgs.cpu())
modelPath = "/Users/niloofar/Documents/Projects/PGD_attacks_TrainMNIST/modelWeights/nonTargeted_model-2.pt"
#torch.save(model.state_dict(), 'Targeted_model_20iter.pt')
model.load_state_dict(torch.load(modelPath, map_location='cpu'))
PGD_Attack.AttackEvaluate(model, TestData)  
