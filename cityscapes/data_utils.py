import os
from PIL import Image

import torch
from torchvision import tv_tensors
from torch.utils.data import Dataset
from torchvision.transforms import v2

class CityScapes(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        assert split in ['train', 'val', 'test'], f"split {split} is not supported. Choose from 'train' or 'val' or 'test'"
        
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
    

def get_transforms(crop_size, norm_mean, norm_std):
    transform_train = v2.Compose([
        v2.ToImage(),
        v2.RandomChoice([
            v2.RandomCrop(size=crop_size),
            v2.RandomResizedCrop(size=crop_size)
        ], [0.7, 0.3]),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomPhotometricDistort(brightness=(0.7, 1.25), contrast=(0.7, 1.25), saturation=(0.7, 1.25), hue=(-0.1, 0.1), p=0.5),
        v2.RandomApply([
            v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
        ], p=0.3),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=norm_mean, std=norm_std),
    ])

    transform_val_test = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=norm_mean, std=norm_std),
    ])

    return transform_train, transform_val_test