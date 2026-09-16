
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np

def build_backbone(name="resnet18", pretrained=True):
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        embedding_dim = 512
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        embedding_dim = 2048
    else:
        raise ValueError("Backbone no soportado.")
    net.fc = nn.Identity()
    return net, embedding_dim

class ProxyModel(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True):
        super().__init__()
        self.backbone, self.embedding_dim = build_backbone(backbone_name, pretrained)
    def forward(self, x):
        return self.backbone(x)
    def load_partial_checkpoint(self, checkpoint_path, device="cpu"):
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.backbone.load_state_dict(state_dict["backbone_state_dict"])

@torch.no_grad()
def compute_embeddings(proxy_model, dataset, transform, device, batch_size=64,
                        num_workers=0, normalize=True, verbose=True):
    from data_utils import SingleViewDataset
    proxy_model.eval()
    proxy_model.to(device)
    wrapped = SingleViewDataset(dataset, transform)
    loader = DataLoader(wrapped, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    all_embeddings = []
    all_labels = []
    total_batches = len(loader)
    for batch_idx, (imgs, labels, idxs) in enumerate(loader):
        imgs = imgs.to(device)
        emb = proxy_model(imgs)
        if normalize:
            emb = nn.functional.normalize(emb, dim=1, p=2)
        all_embeddings.append(emb.cpu().numpy())
        all_labels.append(labels.numpy() if torch.is_tensor(labels) else np.array(labels))
        if verbose and (batch_idx % 10 == 0 or batch_idx == total_batches - 1):
            print(f"  [proxy] batch {batch_idx+1}/{total_batches}")
    embeddings = np.concatenate(all_embeddings, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    return embeddings, labels_arr
