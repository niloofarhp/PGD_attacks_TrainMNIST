import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from AdversarialTrain import AdversarialPGD_Attack

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('/Users/niloofar/Documents/Projects/PGD_attacks_TrainMNIST/Dataset/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            ])),
batch_size=1, shuffle=True)

def AttackEvaluate(model, dataLoader):
    eps = [0, 0.1, 0.2, 0.3, 0.45]
    alpha = 0.01
    pgdAttack = AdversarialPGD_Attack()
    attackType = 'targeted'
    opt = optim.SGD(model.parameters(), lr=1e-2)
    model.eval()
    epochs = 100
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
model = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(7*7*64, 100), nn.ReLU(),
                        nn.Linear(100, 10)).to(device)
modelPath = "/Users/niloofar/Documents/Projects/PGD_attacks_TrainMNIST/modelWeights/nonTargeted_model_20iter.pt"
model.load_state_dict(torch.load(modelPath))
AttackEvaluate(model, test_loader)