import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class ShipDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.images = image_files      # Image Directory
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname       = self.images[idx]
        image       = Image.open(fname).convert('L')


        #'''
        basename = os.path.basename(fname)
        sid         = 0 # '0_xxxx.png'
        eid         = basename.index("_") # '0_xxxx.png'
        label       = basename[sid:eid]
        label       = int(label)
        #'''

        #To get class after first _ notation
        '''
        sid   = fname.index("_")+1 # '0_xxxx.png'
        tmp   = fname[sid:]
        eid   = tmp.index("_")+sid
        label = fname[sid:eid]
        label = int(label)
        '''
        
            
        if self.transform is not None:
            image = self.transform(image)
            #image = augmentations["image"]
        return image, label
