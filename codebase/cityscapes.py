import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors


class CityScapes(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        assert split in ['train', 'val', 'test'], "split should be one of ['train', 'val', 'test']"
        
        self.file_names = []
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'leftImg8bit', split)
        self.targets_dir = os.path.join(root_dir, 'generated_gt', split)

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            tgt_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                if file_name.endswith('_leftImg8bit.png'):
                    img_path = os.path.join(img_dir, file_name)
                    tgt_name = file_name.replace('_leftImg8bit.png', '_trainId.png')
                    tgt_path = os.path.join(tgt_dir, tgt_name)
                    self.file_names.append((img_path, tgt_path))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.file_names[idx][0]
        mask_path = self.file_names[idx][1]
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = tv_tensors.Mask(mask, dtype=torch.long)
        if self.transform:
            image, mask = self.transform((image, mask))

        return image, mask