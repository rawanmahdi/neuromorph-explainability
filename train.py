#%%
import numpy as np
import torch
# from torch.utils.data import DataLoader
from torchvision import transforms
import snntorch as snn
from model.eeg_dataset import EEGDataset, ToSpikes, Reshape
from model.snn import SNN
import time 
#%%
net = SNN(channels=21, time_steps=256, hidden=50, beta=0.9)

loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

num_epochs = 1
counter = 0
#%%
def load_dataset(num_to_load):
    annotations = '../data/annotations/annotations_2017_C.csv'
    data_dir = '../data/neonatal-eeg-edf'
    Dataset = EEGDataset(annotations_file=annotations, eeg_dir=data_dir, transform=transforms.Compose([ToSpikes(1e-6),Reshape()]))
    training_samples = []

    for i in range(1,num_to_load+1):
        training_samples.append(Dataset[i])

    return np.array(training_samples)

#%%
train_ds = load_dataset(3)
# train_loader = DataLoader(train_ds, batch_size=512, shuffle=False)

#%%
device = torch.device('cpu')
def train_snn(model, train_ds, num_epochs):
    start = time.time()
    counter = 0
    model.train()
    loss_history = []
    for epoch in range(num_epochs):
        for sample in train_ds:
            data = sample['eeg']
            labels = sample['annotations']
            # labels = labels[:3]

            spike_rec, mem_rec = model(data)
            
            loss_val = torch.zeros((1))
            loss_val = loss(spike_rec, labels)
            loss_history.append(loss_val.item())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            if counter % 10 == 0:
                print(f"Iterations: {counter} \t Train loss: {loss_val.item()}")

            counter +=1
    end = time.time()
    print(f"Elapsed time in training: {end - start}")
#%%
train_snn(net, train_ds, 1)
# %%
sample = train_ds[0]
data = sample['eeg']
labels = sample['annotations']
spike_rec, mem_rec = net(data)
print(spike_rec.shape, mem_rec.shape)

#%%
print(labels.shape)
#%%
spike_rec = torch.reshape(spike_rec, [spike_rec.shape[0]])
print(spike_rec.shape)