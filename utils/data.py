import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T


def get_transforms(im_size, center_crop,  gray_scale):
    transforms = [T.ToTensor()]

    if center_crop:
        transforms += [T.CenterCrop(size=center_crop)]
    if gray_scale:
        transforms += [T.Grayscale()]
    if im_size is not None:
        transforms += [T.Resize(im_size, antialias=True),]
    transforms+=[T.Normalize((0.5,), (0.5,))]

    return T.Compose(transforms)


class DiskDataset(Dataset):
    def __init__(self, paths, im_size, macro_patch_size=64, center_crop=None, gray_scale=False):
        super(DiskDataset, self).__init__()
        self.paths = paths
        self.transforms = get_transforms(im_size, center_crop, gray_scale)
        self.macro_coord_range = np.arange(0, im_size-macro_patch_size)
        self.im_size = im_size
        self.macro_size = macro_patch_size
        # self.macro_coords = torch.from_numpy(np.stack(np.meshgrid(coord_range, coord_range))[::-1])
        # coord_range = np.arange(micro_size)
        # self.base_micro_coords = torch.from_numpy(np.stack(np.meshgrid(coord_range, coord_range))[::-1])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        y = np.random.randint(0, self.im_size - self.macro_size)
        x = np.random.randint(0, self.im_size - self.macro_size)
        macro_patch = img[..., y:y+self.macro_size, x:x+self.macro_size]
        # return img, idx
        return macro_patch, torch.tensor([x, y])


def get_dataloader(data_root, im_size, batch_size, n_workers, args):
    paths = sorted([os.path.join(data_root, im_name) for im_name in os.listdir(data_root)])
    if args.limit_data is not None:
        paths = paths[:args.limit_data]

    train_dataset = DiskDataset(paths=paths, im_size=im_size, gray_scale=args.gray_scale,
                                center_crop=args.center_crop, macro_patch_size=args.macro_patch_size)
    drop_last = (not args.limit_data) or (args.limit_data != batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,
                              num_workers=n_workers,
                              pin_memory=True, drop_last=drop_last)

    return train_loader
