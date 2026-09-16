
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np

def get_ssl_augmentation(image_size=128):
    return T.Compose([T.RandomResizedCrop(image_size, scale=(0.2,1.0)), T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8), T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(kernel_size=(3,3), sigma=(0.1,2.0))], p=0.5),
        T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def get_eval_transform(image_size=128):
    return T.Compose([T.Resize(int(image_size*1.15)), T.CenterCrop(image_size), T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def get_linear_train_transform(image_size=128):
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class TwoViewDataset(Dataset):
    def __init__(self, base_dataset, augmentation):
        self.base_dataset=base_dataset; self.augmentation=augmentation
    def __len__(self): return len(self.base_dataset)
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        return self.augmentation(img), self.augmentation(img), label, idx

class SingleViewDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset=base_dataset; self.transform=transform
    def __len__(self): return len(self.base_dataset)
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        return self.transform(img), label, idx

def load_food101(cfg, split="train"):
    dataset = torchvision.datasets.Food101(root=cfg.data_root, split=split, download=True)
    if cfg.mode == "debug" and cfg.debug_fraction < 1.0:
        n_total = len(dataset)
        n_keep = max(1, int(n_total * cfg.debug_fraction))
        step = max(1, n_total // n_keep)
        indices = list(range(0, n_total, step))[:n_keep]
        dataset = Subset(dataset, indices)
    return dataset

class DummyImageDataset(Dataset):
    def __init__(self, n_samples=500, n_classes=10, image_size=128, seed=0):
        rng = np.random.default_rng(seed)
        self.n_samples=n_samples; self.n_classes=n_classes; self.image_size=image_size
        self.labels = rng.integers(0, n_classes, size=n_samples)
    def __len__(self): return self.n_samples
    def __getitem__(self, idx):
        rng = np.random.default_rng(idx)
        arr = rng.integers(0,255,size=(self.image_size,self.image_size,3),dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB"), int(self.labels[idx])

def generar_dataset_dummy(cfg):
    if cfg.mode == "debug":
        n_samples = max(50, int(2000 * cfg.debug_fraction))
        n_classes_dummy = min(10, cfg.num_classes)
    else:
        n_samples = 20000
        n_classes_dummy = cfg.num_classes
    return DummyImageDataset(n_samples=n_samples, n_classes=n_classes_dummy, image_size=cfg.image_size)

def build_subset_loader(base_dataset, indices, augmentation, batch_size, num_workers=0, shuffle=True, two_view=True):
    subset = Subset(base_dataset, indices)
    wrapped = TwoViewDataset(subset, augmentation) if two_view else SingleViewDataset(subset, augmentation)
    return DataLoader(wrapped, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
