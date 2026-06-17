import os
import random
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF


class SolarLayoutDataset(Dataset):
    ROOF_TYPES = ['TIN', 'TILE', 'FLAT', 'OTHER']
    CONNECTION_TYPES = ['single_phase', 'three_phase']

    def __init__(self, metadata_csv, image_dir, mask_dir, img_size=(512, 512), augment=True):
        self.df = pd.read_csv(metadata_csv)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.augment = augment

        self.panel_max = float(self.df['num_panels'].max())
        self.strings_max = float(self.df['num_strings'].max())

        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    
    @property
    def metadata_dim(self):
        return 2 + len(self.ROOF_TYPES) + len(self.CONNECTION_TYPES)

    def __encode_metadata(self, row):
        continuous = [
            row['num_panels'] / self.panel_max,
            row['num_strings'] / self.strings_max,
        ]
        roof_onehot = [float(row['roof_type'].strip().lower() == rt) for rt in self.ROOF_TYPES]

        conn_onehot = [float(row['connection_type'].strip().lower() == ct) for ct in self.CONNECTION_TYPES]
        return torch.tensor(continuous + roof_onehot + conn_onehot, dtype=torch.float32)

    def _augment_pair(self, img, mask):
        if random.random() > 0.5:
            img, mask = TF.hflip(img), TF.hflip(mask)
        if random.random() > 0.5:
            img, mask = TF.vflip(img), TF.vflip(mask)
        angle = random.choice([0, 90, 180, 270])
        if angle:
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle)
        img = transforms.ColorJitter(brightness=0.2, contrast=0.2)(img)
        return img, mask
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        roof = Image.open(
            os.path.join(self.image_dir, row['image_filename'])
        ).convert('RGB')
        mask = Image.open(
            os.path.join(self.mask_dir, row['mask_filename'])
        ).convert('L')

        if self.augment:
            roof, mask = self.__augment_pair(roof, mask)
        
        roof_t = self.img_transform(roof)
        mask_t = TF.resize(TF.to_tensor(mask), list(self.img_size), interpolation=TF.InterpolationMode.NEAREST)
        mask_t = (mask_t > 0.5).float()

        return {
            'roof': roof_t,     # (3, H, W)
            'mask': mask_t,     # (1, H, W)
            'metadata': self.__encode_metadata(row), #(meta_dim,)
            'num_panels': int(row['num_panels']),
        }

def build_dataloaders(cfg):
    ds = SolarLayoutDataset(
        cfg['data']['metadata_csv'],
        cfg['data']['image_dir'],
        cfg['data']['mask_dir'],
        img_size=tuple(cfg['data']['img_size']),
        augment=True,
    )
    n_val = max(1, int(len(ds) * cfg['data']['val_split']))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    val_ds.dataset.augment = False

    kwargs = dict(batch_size = cfg['training']['batch_size'],
                  num_workers=cfg['data']['num_workers'],
                  pin_memory=True)
    return (DataLoader(train_ds, shuffle=True, **kwargs),
            DataLoader(val_ds, shuffle=False, **kwargs),
            ds.metadata_dim)