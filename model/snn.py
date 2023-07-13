#%%
import sys
import torch
import snntorch as snn
from snntorch import surrogate
import numpy as np
class SNN(torch.nn.Module):
    def __init__(self, channels, time_steps, hidden, beta):
        super().__init__()

        self.channels = channels
        self.time_steps = time_steps
        self.hidden = hidden
        self.beta = beta

        # Initialize layers
        # layer 1 
        self.fc1 = torch.nn.Linear(in_features=channels, out_features=hidden)
        self.rlif1 = snn.RLeaky(beta=self.beta, all_to_all=True, linear_features=hidden)
        # layer 2
        self.fc2 = torch.nn.Linear(in_features=hidden, out_features=1)
        self.rlif2 = snn.RLeaky(beta=self.beta, all_to_all=True, linear_features=1)
        

    def forward(self, x):
        # Initialize hidden states and outputs at t=0
        spk1, mem1 = self.rlif1.init_rleaky()
        spk2, mem2 = self.rlif2.init_rleaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        num_samples = x.shape[2]

        for i in range(num_samples):
            sample = x[:,:,i]
            for j in range(self.time_steps):
                input = sample[:,j]
                spk1, mem1 = self.rlif1(self.fc1(input), spk1, mem1)
                spk2, mem2 = self.rlif2(self.fc2(spk1), spk2,  mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        return torch.squeeze(torch.stack(spk2_rec)), torch.stack(mem2_rec)
    """
    def __init__(self):
        super().__init__()

        # Parameters, note that some, like threshold, beta, and num_hidden can be learnt 
        num_in = 21
        num_hidden = 1000
        num_out = 1
        beta = 0.95

        # Layers
        self.fc1 = torch.nn.Linear(num_in, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = torch.nn.Linear(num_hidden, num_out)
        self.lif2 = snn.Leaky(beta=beta)
     
    def forward(self, x):
        
        Args:
            x is a sample from the dataset of size [21, 256, X]
        
        
        # Initialize membrain potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record final output spikes and membrane potentials
        spike2_rec = []
        mem2_rec = []

        # Time loop -  instead of passing in the same x each time step, pass in the time varying signals
        num_steps = x.shape[2] # no. seconds per individual
        for step in range(num_steps):
            x_step = x[:,step]
            cur1 = self.fc1(x_step)
            spike1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spike1)
            spike2, mem2 = self.lif2(cur2, mem2)
            spike2_rec.append(spike2)
            mem2_rec.append(mem2)
        return torch.stack(spike2_rec, dim=0), torch.stack(mem2_rec, dim=0)
    """
# %%
