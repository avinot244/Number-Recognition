import torch
from tqdm import tqdm
from globals import *
from utils import *

class MLPModel(torch.nn.Module):
    def __init__(self, activation_fn,
                       n_hidden_layers=2,
                       hidden_layer_size=100,
                       final_activation_fn=torch.nn.Identity()):
        super(MLPModel, self).__init__()

        
        self.inputLayer = torch.nn.Linear(784, hidden_layer_size)

        f1 = lambda x : torch.nn.Linear(hidden_layer_size, hidden_layer_size)
        f2 = lambda x : activation_fn
        
        self.hiddenLayers = [f(_) for _ in range(n_hidden_layers) for f in (f1,f2)]
        self.finalLayer = torch.nn.Linear(hidden_layer_size, 10)
        self.outputLayer = final_activation_fn

        self.allLayers = torch.nn.Sequential(self.inputLayer, *self.hiddenLayers, self.finalLayer, self.outputLayer)
    
    def forward(self, x):
        return self.allLayers(x)

def train_mlp(model, epoch, dataset, loss_fn, optimizer, n_iteration):
    costTbl = []
    count_tbl = []
    c = 0
    print("Epoch  : {}".format(epoch))
    for j in tqdm(range(len(dataset)), ncols=75):
        (x, y) = dataset[j]
        optimizer.zero_grad()
        y_pred = model(x)
        cost = loss_fn(y_pred, y)
        cost.backward()
        optimizer.step()
        if j%(500) == 0:
            c += 1000 + (epoch-1)*len(dataset)
            print("Cost at iteration %i = %f" %(epoch, cost.detach().item()), end="")
            costTbl.append(cost.item())
            count_tbl.append(c)
        
    return costTbl, count_tbl

def test_mlp(model, test_dataset, loss_fn):
    print("\nTesting neural network...")
    test_loss = 0
    for j in tqdm(range(len(test_dataset)), ncols=75):
        (data, ground_truth) = test_dataset[j]
        output = model(data)
        test_loss += loss_fn(output, ground_truth).item()
    test_loss /= len(test_dataset)
    print('Test set: Avg. loss: {:.4f}'.format(test_loss))
    return test_loss

def fit_mlp(tbl_test_losses, tbl_train_losses, tbl_train_counter, model, train_set, test_set, optimizer, loss_fn, n_iteration):
    #Computing initial loss on test dataset without any training
    test_losses = test_mlp(model, test_set, loss_fn)
    tbl_test_losses.append(test_losses)
    for epoch in range(1, n_iteration + 1):
        print("\n",end="")
        #Training
        (train_losses, train_counter) = train_mlp(model, epoch, train_set, loss_fn, optimizer, n_iteration)
        tbl_train_losses.append(train_losses)
        tbl_train_counter.append(train_counter)

        #Test
        test_losses = test_mlp(model, test_set, loss_fn)
        tbl_test_losses.append(test_losses)
    s = getShape(tbl_train_counter)
    tbl_train_counter = reshapeFromShape(tbl_train_counter, s)

    s = getShape(tbl_train_losses)
    tbl_train_losses = reshapeFromShape(tbl_train_losses, s)
    
    return (tbl_train_counter, tbl_train_losses, tbl_test_losses)