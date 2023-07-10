import torch
import pandas as pd
import mne
import os
from torch.utils.data import Dataset
from snntorch import spikegen
from torchvision import transforms

class EEG_Dataset(Dataset):
    def __init__(self, annotations_file, eeg_dir, transform):
        self.annotations = pd.read_csv(annotations_file)
        self.eeg_dir = eeg_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Retrieve individual's EEG reading and array of labels

        Args:
            int: idx

        Returns: 
        
            torch.tensor, torch.tensor: EEG signals across X channels, label for every 256 timesteps
        """
        eeg_path = os.path.join(self.eeg_dir, 'eeg'+str(idx)+'.edf')
        raw_eeg = mne.io.read_raw_edf(eeg_path)
        eeg = torch.tensor(raw_eeg.get_data())
        annotations = torch.tensor(self.annotations[str(idx)].dropna().values)
        sample = {'eeg': eeg, 'annotations':annotations}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class ToSpikes(object):
    """ Encode eeg data into temporal spikes using snntorch's spikgen 
    """
    def __init__(self, threshold):
        """
        Args:
            float: threshold value for difference between two consequtive time steps to generate spike
       
        """ 
        assert isinstance(threshold, float) 
        self.threshold = threshold

    def __call__(self, sample):
        """Implement snntorch.spikegen.delta() to encode time series into spikes
        
        Args:

        Returns:
            list(torch.tensors) : list of encoded spike tensors 
        """
        eeg = sample['eeg']
        encoded_eeg = spikegen.delta(eeg, self.threshold)
        return {'eeg': encoded_eeg, 'annotations': sample['annotations']}

class Reshape(object):
    def __call__(self, sample):
        eeg, annotations = sample['eeg'], sample['annotations']
        eeg = torch.reshape(eeg, (eeg.shape[0], 256, int(eeg.shape[1]/256)))
        return {'eeg': eeg, 'annotations': annotations}


