# exec : python -W ignore main.py

from mlp_model import *
from cnn_model import *
from own_cnn import *
from dataImgReader import ImgDataset
from dataImgReader import getData
from utils import *
from globals import *

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import getopt, sys

def running_mlp_model():
    n_iteration = n_epochs
    loss = torch.nn.BCELoss()
    mlp_model = MLPModel(torch.nn.Sigmoid(), n_hidden_layers=2, final_activation_fn=torch.nn.Sigmoid(), hidden_layer_size=500)
    optimizer_mlp = torch.optim.SGD(mlp_model.parameters(), lr = learning_rate)

    print("Getting train data")
    mlp_dataset_train = ImgDataset(choice="train")
    print("Getting test data")
    mlp_dataset_test = ImgDataset(choice="test")


    
    (tbl_train_counter, tbl_train_losses, tbl_test_losses) = fit_mlp([], [], [], mlp_model, mlp_dataset_train, mlp_dataset_test, optimizer_mlp, loss, n_iteration)
    print(len(tbl_train_counter))
    idx_test_counter = (len(tbl_train_counter)-1)//n_iteration
    test_counter = [tbl_train_counter[idx_test_counter*i] for i in range(n_iteration + 1)]
    print(test_counter)

    # Printing loss graph of the training tested on the test set    
    fig = plt.figure()
    plt.plot(tbl_train_counter, tbl_train_losses, color = 'blue')
    plt.scatter(test_counter, tbl_test_losses, color ='red')
    plt.legend(['Train Loss', 'Test Loss'], loc = 'upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    (_, test_set) = getData()
    examples = enumerate(test_set)
    _, (example_data, example_targets) = next(examples)
    
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        (x, _) = mlp_dataset_test[i]
        pred_vector = mlp_model(x)
        pred = getPrediction(pred_vector)
        plt.title("Prediction: {}".format(pred))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def running_cnn_model():
    torch.manual_seed(random_seed)
    (train_set, test_set) = getData()
    cnn_model = CNNModel()
    test_counter = [i*len(train_set.dataset) for i in range (n_epochs + 1)]

    optimizer_cnn = torch.optim.SGD(cnn_model.parameters(), lr = learning_rate)
    
    for param in cnn_model.parameters():
        print(type(param), param.size())

    (tbl_train_counter, tbl_train_losses, tbl_test_losses) = fit_cnn([], [], [], cnn_model, train_set, test_set, optimizer_cnn)

    # Printing loss graph of the training tested on the test set    
    plt.plot(tbl_train_counter, tbl_train_losses, color = 'blue')
    plt.scatter(test_counter, tbl_test_losses, color ='red')
    plt.legend(['Train Loss', 'Test Loss'], loc = 'upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    examples = enumerate(train_set)
    _, (example_data, _) = next(examples)
    # shape of data : (64, 1, 28, 28)
    # we have a batch size of 64 and img size of 28x28
    # data in example_data[batch_idx][i]
    print(example_data[0][0].size())
    # shape of ground_truth : (64)
    # each ith element has an integer that correspond to the ground truth of the example_data[i] img

    print(type(example_data))

    print("All Predictions :")
    with torch.no_grad():
        output = cnn_model(example_data)
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
    print("Predictions from the first convolutional layer")
    # plt.imshow(cnn_model.getFirstFilter())
    # plt.imshow(cnn_model.applyFirstFilter(example_data), cmap='gray', interpolation='none')
    #use torch.nn.functional.conv2D

def running_own_cnn_model():
    torch.manual_seed(random_seed)
    (train_set, test_set) = getData()
    cnn_model = OwnCNNModel()
    test_counter = [i*len(train_set.dataset) for i in range (n_epochs + 1)]

    optimizer_cnn = torch.optim.SGD(cnn_model.parameters(), lr = learning_rate)
    
    for param in cnn_model.parameters():
        print(type(param), param.size())

    (tbl_train_counter, tbl_train_losses, tbl_test_losses) = fit_cnn([], [], [], cnn_model, train_set, test_set, optimizer_cnn)

    # Printing loss graph of the training tested on the test set    
    plt.plot(tbl_train_counter, tbl_train_losses, color = 'blue')
    plt.scatter(test_counter, tbl_test_losses, color ='red')
    plt.legend(['Train Loss', 'Test Loss'], loc = 'upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    examples = enumerate(train_set)
    _, (example_data, _) = next(examples)
    # shape of data : (64, 1, 28, 28)
    # we have a batch size of 64 and img size of 28x28
    # data in example_data[batch_idx][i]
    print(example_data[0][0].size())
    # shape of ground_truth : (64)
    # each ith element has an integer that correspond to the ground truth of the example_data[i] img

    print(type(example_data))

    print("All Predictions :")
    with torch.no_grad():
        output = cnn_model(example_data)
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
if __name__ == "__main__":
    argumentList = sys.argv[1:]
    options = "hmo:"
    lst_options = ["help", "model="]
    try :
        arguments, values = getopt.getopt(argumentList, options, lst_options)

        for currentArgument, currentValue in arguments:
            if currentArgument in ("-h", "--help"):
                print("Help")
            elif currentArgument in ("-m", "--model"):
                print(("Enabling model (% s)") % (currentValue))
                if currentValue == "mlp":
                    running_mlp_model()
                elif currentValue == "cnn":
                    running_cnn_model()
                elif currentValue == "owncnn":
                    running_own_cnn_model()
    except getopt.error as err:
        print(str(err))
