import torch
import torch.nn.functional as F
from globals import *
from tqdm import tqdm
from utils import *

class OwnCNNModel(torch.nn.Module):
    def __init__(self) -> None:
        super(OwnCNNModel, self).__init__()
        #conv-relu-pool x2
        self.conv1 = torch.nn.Conv2d(1, 15, kernel_size=7)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(15, 7, kernel_size=3)
        self.fc1 = torch.nn.Linear(7 * 4 * 4, 120)
        self.fc2 = torch.nn.Linear(120, 50)
        self.fc3 = torch.nn.LayerNorm(50,10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 7 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)
def train_cnn(model, epoch, train_dataset, optimizer):
    model.train()
    train_losses = []
    train_counter = []
    for batch_idx, (data, ground_truth) in enumerate(train_dataset):
        optimizer.zero_grad()
        pred = model(data)
        loss = F.nll_loss(pred, ground_truth)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{:5.0f}/{:5.0f} ({:2.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataset.dataset), 
                100. * batch_idx / len(train_dataset), loss.item()), end='\r')
            train_losses.append(loss.item())
            train_counter.append(
               (batch_idx*64) + ((epoch-1)*len(train_dataset.dataset)))
            
            torch.save(model.state_dict(), './results/cnn/model.pth')
            torch.save(optimizer.state_dict(), './results/cnn/optimize.pth')
    print("", end= "\n")
    return (train_losses, train_counter)
        
def test_cnn(model, test_dataset):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, ground_truth in tqdm(test_dataset, ncols = 100):
            output = model(data)
            test_loss += F.nll_loss(output, ground_truth, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(ground_truth.data.view_as(pred)).sum()
    test_loss /= len(test_dataset.dataset)
    print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataset.dataset),
        100.*correct/len(test_dataset.dataset)), end="")
    return test_loss

def fit_cnn(tbl_test_losses, tbl_train_losses, tbl_train_counter, model, train_set, test_set, optimizer):
    #Computing initial loss on test dataset without any training
    test_losses = test_cnn(model, test_set)
    tbl_test_losses.append(test_losses)
    for epoch in range(1, n_epochs + 1):
        #Training
        (train_losses, train_counter) = train_cnn(model, epoch, train_set, optimizer)
        tbl_train_losses.append(train_losses)
        tbl_train_counter.append(train_counter)

        #Test
        test_losses = test_cnn(model, test_set)
        tbl_test_losses.append(test_losses)
    s = getShape(tbl_train_counter)
    tbl_train_counter = reshapeFromShape(tbl_train_counter, s)

    s = getShape(tbl_train_losses)
    tbl_train_losses = reshapeFromShape(tbl_train_losses, s)
    return (tbl_train_counter, tbl_train_losses, tbl_test_losses)
    