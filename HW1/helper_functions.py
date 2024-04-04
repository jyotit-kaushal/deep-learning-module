# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Numpy
import numpy as np
# Pandas
import pandas as pd
# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def load_dataset(excel_file_path = 'dataset.xlsx'):
    df = pd.read_excel(excel_file_path)
    x1 = df['x1'].values
    x2 = df['x2'].values
    y = df['class'].values
    x = np.column_stack([x1, x2])
    return x1, x2, x, y

def plot_dataset(min_val, max_val, train_val1_list, train_val2_list, train_outputs):
    # Initialize plot
    fig = plt.figure(figsize = (10, 7))

    # Scatter plot
    markers = {0: "x", 1: "o", 2: "P"}
    colors = {0: "r", 1: "g", 2: "b"}
    indexes_0 = np.where(train_outputs == 0)[0]
    v1_0 = train_val1_list[indexes_0]
    v2_0 = train_val2_list[indexes_0]
    indexes_1 = np.where(train_outputs == 1)[0]
    v1_1 = train_val1_list[indexes_1]
    v2_1 = train_val2_list[indexes_1]
    indexes_2 = np.where(train_outputs == 2)[0]
    v1_2 = train_val1_list[indexes_2]
    v2_2 = train_val2_list[indexes_2]
    plt.scatter(v1_0, v2_0, c = colors[0], marker = markers[0])
    plt.scatter(v1_1, v2_1, c = colors[1], marker = markers[1])
    plt.scatter(v1_2, v2_2, c = colors[2], marker = markers[2])


# Test Function for Dataset object
def test_dataset_oject(dataset):
    # Test case 1
    print("--- Test case 1 (dataset): Implemented the correct __getitem__ method.")
    print("Sample with index 384 was drawn from dataset.")
    index = 384
    true1 = (torch.tensor([0.9700, 0.7500]), torch.tensor(0.))
    try:
        test1 = dataset[index]
    except:
        test1 = "Something went wrong when retrieving sample 384."
    print("Retrieved: {}".format(test1))
    print("Expected: {}".format(true1))
    try:
        c1 = (test1[0] == true1[0]).all()
        c2 = (test1[1] == true1[1])
        val = "Passed" if c1 and c2 else "Failed"
    except:
        val = "Failed"
    print("Test case 1: {}".format(val))
    
    # Test case 2
    print("--- Test case 2 (dataset): Implemented the correct __len__ method.")
    print("Asking for dataset length.")
    true2 = 1024
    try:
        test2 = len(dataset)
    except:
        test2 = "Something went wrong when asking for dataset length."
    print("Retrieved: {}".format(test2))
    print("Expected: {}".format(true2))
    try:
        test2 = len(dataset)
        val = "Passed" if test2 == true2 else "Failed"
    except:
        val = "Failed"
    print("Test case 2: {}".format(val))


# Test Function for Dataloader object
def test_dataloader_oject(dataloader):
    # Test case 1
    print("--- Test case 1 (dataloader): Using the correct batch size.")
    print("Asking for batch size.")
    true1 = 128
    try:
        test1 = dataloader.batch_size
    except:
        test1 = "Something went wrong when checking batch size."
    print("Retrieved: {}".format(test1))
    print("Expected: {}".format(true1))
    try:
        val = "Passed" if test1 == true1 else "Failed"
    except:
        val = "Failed"
    print("Test case 1: {}".format(val))
    
    # Test case 2
    print("--- Test case 2 (dataloader): Dataloader is shuffling the dataset, as requested.")
    print("Asking if Dataloader will be shuffling.")
    true2 = True
    try:
        test2 = "torch.utils.data.sampler.RandomSampler object at" in str(dataloader.sampler)
    except:
        test2 = "Something went wrong when checking shuffling."
    print("Retrieved: {}".format(test2))
    print("Expected: {}".format(true2))
    try:
        val = "Passed" if test2 == true2 else "Failed"
    except:
        val = "Failed"
    print("Test case 2: {}".format(val))


# Test Function for WeirdActivation object
def test_act_oject(act_fun, device):
    # Test case 1
    print("--- Test case (activation function): Checking for correct forward method.")
    print("Testing forward on a Tensor of values.")
    true2 = [[1. ], [-0.2], [0. ]]
    try:
        x = torch.tensor([[1],[-1],[0]], device = device)
        #act_fun.a = 0.2
        out = act_fun.forward(x).cpu().detach().numpy()
        test2 = [[round(float(out[0][0]), 3)], [round(float(out[1][0]), 3)], [round(float(out[2][0]), 3)]]
        print(test2)
    except:
        test2 = "Something went wrong when checking forward method."
    print("Retrieved: {}".format(test2))
    print("Expected: {}".format(true2))
    try:
        val = "Passed" if test2 == true2 else "Failed"
    except:
        val = "Failed"
    print("Test case: {}".format(val))
        

def check_activation_curves(fun_obj, device):
    # Define parameter values
    a_values = [0, 0.2, 0.5, 1]
    
    # Create activation function instances for each parameter value
    activations = [fun_obj(a, device) for a in a_values]
    
    # Generate x values for plotting
    x_values = np.linspace(-5, 5, 1000)
    x_tensor = torch.tensor(x_values, dtype = torch.float64, device = device)
    
    # Plot activation functions
    plt.figure(figsize = (10, 8))
    for i, activation in enumerate(activations, 1):
        y_values = activation(x_tensor).cpu().detach().numpy()
        plt.subplot(2, 2, i)
        plt.plot(x_values, y_values)
        plt.title(f'a = {a_values[i-1]}')
        plt.xlabel('x')
        plt.ylabel('Activation')
    plt.suptitle("WeirdActivation function curves for different values of a = {0, 0.2, 0.5, 1}")
    plt.tight_layout()
    plt.show()