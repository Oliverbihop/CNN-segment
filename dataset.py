import torch.utils.data as data
from PIL import Image
import numpy as np
import glob
import os
from torchvision import transforms


class DataLoaderSegmentation(data.Dataset):
    def __init__(self, folder_path,transform=None,image_size=[48, 48]):
        super(DataLoaderSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'image','*.jpg'))
        self.mask_files = []
        self.image_size = image_size
        self.transform= transform
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path,'mask',os.path.basename(img_path)))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            _img = Image.open(img_path).convert('RGB')
            _tmp = np.array(Image.open(mask_path).convert('L'), dtype=np.uint8)
            _target = self.encode_segmap(_tmp)
            sample={'image': _img,'label': _target}
            if self.transform:
                sample = self.transform(sample)
            return sample #torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        return len(self.img_files)
    def encode_segmap(self, mask):
        # Put all void classes to ignore_index
        mask[mask!=0]=1
        mask=Image.fromarray(mask)
        return mask