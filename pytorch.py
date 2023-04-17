import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from tabular_data import hello
from sklearn.preprocessing import scale
h = hello()
import warnings
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from tabular_data import hello
import re
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
from sklearn.metrics import r2_score
import json
import os
import itertools

class AirbnbNightlyPriceImageDataset(Dataset):

    def __init__(self):
        super().__init__()
        # Load the dataset from a CSV file
        self.data = pd.read_csv("/Users/dq/Documents/aicore_project/Airbnb_Project/clean_tabular_data.csv")
        # Remove the 'Unnamed: 0' column
        self.data.pop("Unnamed: 0")
        # Drop the 'Price_Night' column and store the remaining features in a array
        self.features = self.data.drop(['Price_Night'], axis=1)
        self.features = self.features.to_numpy(dtype=float)
        # Scale the features
        self.features = scale(self.features)
        # Store the 'Price_Night' column as the target variable and scale it
        self.labels = self.data['Price_Night']
        self.labels = self.labels.to_numpy(dtype=float)
        self.labels = scale(self.labels)
        h = hello()

    def __getitem__(self, index):
        # Get the features and label
        features = self.features[index,:]
        features=torch.tensor(features).unsqueeze(0)
        label= self.labels[index]
        label = torch.tensor(label)
        # Return a tuple of the features and label as tensors
        return(features, label)

    def __len__(self):
        # Return the length of the dataset
        return len(self.data)

# Create an instance of the AirbnbNightlyPriceImageDataset class
dataset = AirbnbNightlyPriceImageDataset()
features, labels = dataset[1]
# print(labels)

# Split the dataset into training, testing, and validation subsets
tr, test = random_split(dataset, [round(len(dataset)*0.8), round(len(dataset)*0.2)])
tr, valid = random_split(dataset, [round(len(dataset)*0.75), round(len(dataset)*0.25)])

# Create data loaders for the training, testing, and validation subsets
train_loader = DataLoader(tr, batch_size=16, shuffle=True)
test_loader = DataLoader(test, batch_size=16)
valid_loader = DataLoader(valid, batch_size=16)

# Get the features and labels for the first batch of data from the training data loader
features, labels = next(iter(train_loader))
# print(features.shape)

import yaml
def get_nn_config():
    # Open the 'nn_config.yml' file in read mode
    with open("/Users/dq/Documents/aicore_project/Airbnb_Project/nn_config.yml", "r") as stream:
        # Load the contents of the file into a dictionary using the 'yaml.safe_load' function
        dict = yaml.safe_load(stream)
    
    return dict

class LinearRegression(torch.nn.Module):
    def __init__(self)->None:
        super().__init__()
        # hidden layers - add more complexity to make it more accurate
        # if no layers (activation functions), the model will be reduced to a simple linear model

        # Define a sequence of linear layers and activation functions to add complexity to the model
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(11, 15),
            torch.nn.ReLU(),
            torch.nn.Linear(15, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 1)
        )

        # Convert the model parameters to floating point numbers
        self.double()
    
    # Define how the input features are passed through the layers of the model to generate a predicted label
    def forward(self, features):
        # Pass the input features through the linear layers and activation functions (self.layers) to generate a predicted label
        return self.layers(features)

def evaluate(model, dataloader):
    model.eval()
    losses = []
    n_examples = 0
    # iterate over each batch of data in the dataloader
    for batch in dataloader:
        # extract the features and labels from the batch
        features, labels = batch
        # make predictions using the model
        prediction = model(features)
        # calculate mean squared error loss between the predictions and the labels
        loss = F.mse_loss(prediction, labels)
        # append the loss to the list
        losses.append(loss.detach())
        # update the number of examples processed
        n_examples += len(labels)
    # calculate the average loss over all the batches
    avg_loss = np.mean(losses)
    return avg_loss

def train(model, config, epochs=10):
    # Get the optimizer class and instance from the config
    optimiser_class = config['optimiser']
    optimiser_instance = getattr(torch.optim, optimiser_class)
    # Create an instance of the optimizer and set the learning rate from the config
    optimiser = optimiser_instance(model.parameters(), lr=config['learning_rate'])

    # Start the timer for training duration
    start_time = time.time()
    dt_now = datetime.now()

    # Create empty lists to store loss values during training and validation
    loss_list = []
    val_loss = []
    # Create an instance of SummaryWriter to write TensorBoard logs
    # optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    writer=SummaryWriter()

    # Initialize variables for batch index, number of predictions, and RMSE
    batch_idx=0
    num_predictions=0
    rmse=0

    # Loop through the number of epochs specified
    for epoch in range(epochs):

        # train the model
        model.train()

        # Loop through the batches in the training data
        for batch in train_loader:

            # Get the features and labels for the batch
            features, labels = batch

            # Make a prediction for the batch using the model
            prediction = model(features)

            # Calculate the MSE loss for the batch
            loss = F.mse_loss(prediction, labels)

             # Add the loss value to the list of losses
            loss_list.append(loss)
            # Backpropagate the loss through the model
            loss.backward() #
            # print(loss.item())
            # Calculate the RMSE for the batch and add it to the running total
            rmse += torch.sqrt(loss)
            # Take a step in the optimizer to update the model parameters
            optimiser.step()
            # Zero out the gradients to prepare for the next batch
            optimiser.zero_grad()
            # Write the loss value to TensorBoard
            writer.add_scalar('loss', loss.item(), batch_idx)
            # Increment the batch index
            batch_idx+=1
        
        # Calculate the validation loss after each epoch
        val = evaluate(model, valid_loader)
        val_loss.append(val)
        # print(val_loss)
    
    # Concatenate the predictions from all batches and calculate the R-squared value
    prediction_list = np.concatenate([pred.detach().numpy() for pred in prediction])    
    r2 = r2_score(labels, prediction_list)

    # Calculate the total training duration and inference latency per prediction
    training_duration = time.time() - start_time
    inference_latency = (time.time() - start_time) / len(prediction)   
    # rmse = rmse / num_predictions
    # Create a dictionary of metrics
    metrics = {
        "Avg_RMSE_loss": str(rmse),
        "R_squared": r2, 
        "training_duration": training_duration,
        "inference_latency": inference_latency
    }
    return metrics

def save_model(model, config, metrics):

    # Check if the input model is an instance of the PyTorch nn.Module class
    if isinstance(model, torch.nn.Module):
        # Get the hyperparameters
        config = get_nn_config()
        # Train the model for 5 epochs and save the metrics
        metrics = train(model, config, epochs=5)
        # Get the current date and time to use as the model name
        model_name = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
         # Create a folder with the model name
        model_folder = '/Users/dq/Documents/aicore_project/Airbnb_Project/neural_networks/regression/'+model_name
        folder = os.mkdir(model_folder)
        # Save the PyTorch model's state dictionary to a file in the model folder
        torch.save(model.state_dict(),model_folder + '/model.pt')
        
        # Save the hyperparameters used to train the model to a JSON file in the model folder
        with open(f'{model_folder}/hyperparameters.json', 'w') as file:
            json.dump(config, file)

        # Save the training metrics to a JSON file in the model folder
        with open(f'{model_folder}/metrics.json', 'w') as file:
            json.dump(metrics, file)

def train_model(config):
    # Create a new instance of a LinearRegression model
    model = LinearRegression()
    # Get the hyperparameters
    config = get_nn_config()
    # Train the model for 5 epochs and save the metrics
    metrics = train(model, config, epochs=5)
    # Save the model, hyperparameters and metrics to a folder
    d = save_model(model, config, metrics)
    return d

def generate_nn_configs():
    # Get the hyperparameters
    nn_config = get_nn_config()
    # print(nn_config)
    # Get the keys and values of the hyperparameter dictionary
    keys = nn_config.keys()
    vals = nn_config.values()
    # params, values = zip(*nn_config.items())
    # Generate all possible combinations of the values for each key
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*vals)]
    return combos

def find_best_nn(model):
    # Initialize variables
    best_rmse = np.inf
    best_r2 = 0
    number = 1

    # Generate a list of all possible hyperparameter configurations
    hyper_dict_list = generate_nn_configs()
    # Loop over each hyperparameter configuration and train a neural network with those parameters
    for hyper_dict in hyper_dict_list:
        print(f"Now Training model {number}")
        # Model trained using hyperparameters
        metrics = train(model, hyper_dict)
        print(metrics)
        number += 1

        #Checks if the R-squared value of the current trained model is better than the current best R-squared value. 
            #  If so, it updates the best_r2, best_model, best_hyperparams, and best_metrics variables with the values 
            # from the current trained model.
        if metrics['R_squared'] > best_r2:
            print("in If statement now")
            best_r2 = metrics["R_squared"]
            best_model = model
            best_hyperparams = hyper_dict
            best_metrics = metrics
    
    # Print the best model configuration and associated metrics
    print("Best Model:", "\n", best_model, best_r2, best_hyperparams, best_metrics)

    # Check if the input model is an instance of the PyTorch nn.Module class
    if isinstance(model, torch.nn.Module):
        # Get the current date and time to use as the model name
        model_name = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
         # Create a folder with the model name
        model_folder = '/Users/dq/Documents/aicore_project/Airbnb_Project/neural_networks/regression/'+model_name
        folder = os.mkdir(model_folder)

        # Save the PyTorch model's state dictionary to a file in the model folder
        torch.save(best_model.state_dict(),model_folder + '/model.pt')
        
        # Save the hyperparameters used to train the model to a JSON file in the model folder
        with open(f'{model_folder}/hyperparameters.json', 'w') as file:
            json.dump(best_hyperparams, file)

        # Save the training metrics to a JSON file in the model folder
        with open(f'{model_folder}/metrics.json', 'w') as file:
            json.dump(best_metrics, file)

if  __name__ == '__main__':
    warnings.filterwarnings("ignore", message="Using a target size .*")
    # Create a new instance of a LinearRegression model
    model = LinearRegression()
    # Find the best hyperparameter configuration for the linear regression model
    find_best_nn(model)

