#%%
import sys
import torch
import snntorch as snn

class SNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Parameters, note that some, like threshold, beta, and num_hidden can be learnt 
        num_in = 21*256
        num_hidden = 1000
        num_out = 1
        beta = 0.95

        # Layers
        self.fc1 = torch.nn.Linear(num_in, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = torch.nn.Linear(num_hidden, num_out)
        self.lif2 = snn.Leaky(beta=beta)
     
    def forward(self, x):
        """
        Args:
            x is a sample from the dataset of size [21, 256, X]
        
        """
        # Initialize membrain potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record final output spikes and membrane potentials
        spike2_rec = []
        mem2_rec = []

        # Time loop
        num_steps = x.shape[2]
        for step in range(num_steps):
            x_step = x[:,:,step]
            spike2_256 = []
            mem2_256 = []
            
            for point in range(3): # 256 points x.shape[1]
                x_point = x_step[:, point]
                cur1 = self.fc1(x_point)
                spike1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spike1)
                spike2, mem2 = self.lif2(cur2, mem2)
                spike2_256.append(spike2)
                mem2_256.append(mem2)
            
            spike2_rec.append(sum(spike2_256)/256)
            mem2_rec.append(sum(mem2)/256)

        return torch.stack(spike2_rec, dim=0), torch.stack(mem2_rec, dim=0)
