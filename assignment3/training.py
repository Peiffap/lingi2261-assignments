import argparse
import numpy as np
from contest_agent import SmallDeepNetwork
import pandas as pd
import time
from ignite.metrics import Loss as MLoss
from ignite.metrics import Accuracy as MAccuracy
import torch
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def main(path):
        
    # Set the training parameters
    epochs = 1
    lr = 0.5
    batch_size = 10
    
    # Torchvision contains a link to download the FashionMNIST dataset. Let's first 
    # store the training and test sets.
    
    train_data = SquadroDataset(path = 'data/smart_agent_data.csv')
    
    # We now divide the training data in training set and validation set.
    n_train = len(train_data)
    indices = list(range(n_train))
    split = int(n_train - (n_train * 0.1))  # Keep 10% for validation
    train_set = Subset(train_data, indices[:split])
    val_set = Subset(train_data, indices[split:])
    
    # Object where the data should be moved to:
    #   either the CPU memory (aka RAM + cache)
    #   or GPU memory
    #device = torch.device('cuda') # Note: cuda is the name of the technology inside NVIDIA graphic cards
    #network = DeepNetwork().to(device) # Transfer Network on graphic card.
    
    network = SmallDeepNetwork()
    network.load_state_dict(torch.load('model/model1.pt'))
    network.eval()
    
    optimizer = optim.SGD(network.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    print_network(network, optimizer)

    
    # Load the data
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    
    # Metrics in order to check how the training's going
    metrics = MetricsList(
        metrics=[MLoss(F.cross_entropy), MAccuracy()],
        names=["Loss", "Accuracy"]
    )
    
    
    # Complete Training Loop
    for epoch in range(epochs):
        print(f"--- Starting epoch {epoch}")
        start = time.time()
        
        # Train the model
        print("Training...")
        metrics.reset()
    
        network.train() # Set the network in training mode => weights become trainable aka modifiable
        for batch in train_loader:
            x, t = batch
            #x, t = x.to(device), t.to(device)
            x = x.float()
            t = t.long()

            optimizer.zero_grad()  # (Re)Set all the gradients to zero

            network = network.float()
            y, vh = network(x)  # Infer a batch through the network
            y = y.float()
            #print('=============================')
            #print(x.data)
            #print(t.data)
            #print(y.data)
            #print('=============================')
            
            
            loss = criterion(y, t)  # Compute the loss
            loss.backward()  # Compute the backward pass based on the gradients and activations
            optimizer.step()  # Update the weights

            metrics.update(y, t)
        metrics.compute("Train")
    
         # Validate the model
        print("Validating...")
        with torch.no_grad():
            metrics.reset()

            network.eval() # Freeze the network >< training mode
            for batch in val_loader:
                x, t = batch
                #x, t = x.to(device), t.to(device)
    
                y, vh = network(x)
                y = y.float()
                metrics.update(y, t)
            metrics.compute("Validation")
        print(metrics)
        metrics.clear()
        
        end = time.time()
        
        # Print logging
        print(f"\n-Ending epoch {epoch}: elapsed time {end - start}\n")
        
    torch.save(network.state_dict(), 'model/model1.pt')


class MetricsList():
    def __init__(self, metrics, names):
        self.metrics = metrics
        self.names = names
        self.df = pd.DataFrame(columns=names)

    def update(self, logits, labels):
        """
            Updates all the metrics
            - logits: output of the network
            - labels: ground truth
        """
        for metric in self.metrics:
            metric.update((logits, labels))

    def reset(self):
        for metric in self.metrics:
           metric.reset()

    def clear(self):
       self.df = self.df.iloc[0:0]  # Clear Dataframe
    
    def compute(self, mode):
        data = []
        for metric in self.metrics:
           data.append( metric.compute() )
        self.df.loc[mode] = data
    
    def __str__(self):
       return str(self.df)


 
 
class SquadroDataset(Dataset):
    def __init__(self, path):
        # # All the data preperation tasks can be defined here
        # - Deciding the dataset split (train/test/ validate)
        # - Data Transformation methods 
        # - Reading annotation files (CSV/XML etc.)
        # - Prepare the data to read by an index
        
        data = pd.read_csv(path).to_numpy()
        x = data[:,:-1].astype(float)
        t = data[:,-1].astype(int)
        self.x = np.transpose(torch.from_numpy(x)).float()
        self.t = np.transpose(torch.from_numpy(t)).long()
        
         
    def __getitem__(self, index):
        # # Returns data and labels
        # - Apply initiated transformations for data
        # - Push data for GPU memory
        # - better to return the data points as dictionary/ tensor  
        return (self.x[:,index], self.t[index])
 
    def __len__(self):
        return len(self.t)

def print_network(network, optimizer):
    # Print model's state_dict
    mod_dict = network.state_dict()
    print("Model's state_dict:")
    for param_tensor in mod_dict:
        print(param_tensor, "\t", mod_dict[param_tensor].size())
        print(mod_dict[param_tensor])
    
    # Print optimizer's state_dict
    opt_dict = optimizer.state_dict()
    print("Optimizer's state_dict:")
    for var_name in opt_dict:
        print(var_name, "\t", opt_dict[var_name])
        print(opt_dict[var_name])   
        
        

if __name__ == "__main__":
	  parser = argparse.ArgumentParser()
	  parser.add_argument("-p", help="path")
	  args = parser.parse_args()

	  path = args.p if args.p != None else "error"

	  main(path)
