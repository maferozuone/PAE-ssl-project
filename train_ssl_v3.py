"""train_ssl_v3.py
Version de train_ssl_v2 optimizada para el cuello de botella del DataLoader.
Usa config_v3.py.
"""

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config_v3 import get_config
from data_utils import load_food101, generar_dataset_dummy, get_ssl_augmentation, TwoViewDataset
from ssl_models import build_ssl_model
from ssl_losses import simsiam_loss, byol_loss, cpc_loss, align_uniform_loss


def image_to_strips(image_tensor, n_rows=4):
    C, H, W = image_tensor.shape
    if H % n_rows != 0:
        raise ValueError(f"La altura {H} no es divisible entre n_rows={n_rows}")
    h = H // n_rows
    return torch.stack([image_tensor[:, i*h:(i+1)*h, :] for i in range(n_rows)])


class CPCDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform, n_rows=4):
        self.base_dataset = base_dataset
        self.transform = transform
        self.n_rows = n_rows

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        return image_to_strips(self.transform(image), self.n_rows), label, idx


def valid_data_configs():
    return {
        "full",
        "sas_keep_90pct", "sas_keep_80pct", "sas_keep_60pct", "sas_keep_40pct",
        "random_keep_90pct", "random_keep_80pct", "random_keep_60pct", "random_keep_40pct",
    }


def load_dataset_and_indices(cfg, data_config, use_dummy=False):
    if data_config not in valid_data_configs():
        raise ValueError(f"data_config no valido: {data_config}")

    dataset = generar_dataset_dummy(cfg) if use_dummy else load_food101(cfg, split="train")
    if data_config == "full":
        return dataset, None

    suffix = "debug" if use_dummy and cfg.mode == "debug" else "full"
    path = os.path.join(cfg.subset_dir, f"{data_config}_{suffix}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontro el subconjunto: {path}")

    indices = np.load(path).astype(np.int64)
    if len(indices) == 0 or np.min(indices) < 0 or np.max(indices) >= len(dataset):
        raise IndexError(f"Indices invalidos en {path} para dataset de {len(dataset)} muestras")
    return dataset, indices


def get_backbone_state_dict(model, method):
    if method in ("simsiam", "align_uniform"):
        return model.backbone.state_dict()
    if method == "byol":
        return model.online_backbone.state_dict()
    if method == "cpc":
        return model.encoder.backbone.state_dict()
    raise ValueError(f"Metodo no reconocido: {method}")


def training_step(method, model, batch, device):
    logs = {}
    non_blocking = device.type == "cuda"

    if method == "simsiam":
        x1, x2, _, _ = batch
        x1, x2 = x1.to(device, non_blocking=non_blocking), x2.to(device, non_blocking=non_blocking)
        p1, z2, p2, z1 = model(x1, x2)
        loss = simsiam_loss(p1, z2, p2, z1)
    elif method == "byol":
        x1, x2, _, _ = batch
        x1, x2 = x1.to(device, non_blocking=non_blocking), x2.to(device, non_blocking=non_blocking)
        q1, tz2, q2, tz1 = model(x1, x2)
        loss = byol_loss(q1, tz2, q2, tz1)
    elif method == "align_uniform":
        x1, x2, _, _ = batch
        x1, x2 = x1.to(device, non_blocking=non_blocking), x2.to(device, non_blocking=non_blocking)
        z1, z2 = model(x1, x2)
        loss, l_align, l_uniform = align_uniform_loss(z1, z2)
        logs["l_align"] = l_align.item()
        logs["l_uniform"] = l_uniform.item()
    elif method == "cpc":
        strips, _, _ = batch
        strips = strips.to(device, non_blocking=non_blocking)
        z, _, predictions = model(strips)
        loss = cpc_loss(predictions, z)
    else:
        raise ValueError(f"Metodo no reconocido: {method}")
    return loss, logs


def build_loader(wrapped, cfg, n_used):
    kwargs = {
        "batch_size": min(cfg.batch_size, n_used),
        "shuffle": True,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.device.type == "cuda",
        "persistent_workers": cfg.num_workers > 0,
        "drop_last": True,
    }
    if cfg.num_workers > 0 and cfg.prefetch_factor is not None:
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return DataLoader(wrapped, **kwargs)


def train(cfg, method, data_config, use_dummy=False):
    print(f"\n{'='*70}")
    print(f"Entrenando: metodo={method} | datos={data_config} | modo={cfg.mode}")
    print(f"{'='*70}\n")

    dataset, indices = load_dataset_and_indices(cfg, data_config, use_dummy)
    n_used = len(indices) if indices is not None else len(dataset)
    print(f"Dataset base: {len(dataset)} | Usando: {n_used} muestras")
    print(f"Dispositivo: {cfg.device}")
    print(f"batch_size={cfg.batch_size} | num_workers={cfg.num_workers} | pin_memory={cfg.device.type == 'cuda'}")

    base = Subset(dataset, indices.tolist()) if indices is not None else dataset
    transform = get_ssl_augmentation(cfg.image_size)
    wrapped = CPCDataset(base, transform, 4) if method == "cpc" else TwoViewDataset(base, transform)
    loader = build_loader(wrapped, cfg, n_used)

    if len(loader) == 0:
        raise ValueError(f"No hay batches: n_used={n_used}, batch_size={cfg.batch_size}")

    model = build_ssl_model(method, cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.device.type == "cuda" and cfg.use_amp)
    history = []

    for epoch in range(cfg.epochs_ssl):
        start = time.perf_counter()
        model.train()
        losses, extra = [], {}

        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=cfg.device.type, dtype=torch.float16,
                                enabled=cfg.device.type == "cuda" and cfg.use_amp):
                loss, logs = training_step(method, model, batch, cfg.device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if method == "byol":
                model.update_target_network()
            losses.append(float(loss.detach().cpu()))
            for key, value in logs.items():
                extra.setdefault(key, []).append(value)
            if batch_idx % cfg.log_every == 0:
                print(f"  Epoch {epoch+1}/{cfg.epochs_ssl} | Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

        rec = {"epoch": epoch+1, "avg_loss": float(np.mean(losses)),
               "time_sec": time.perf_counter()-start}
        for key, values in extra.items():
            rec[f"avg_{key}"] = float(np.mean(values))
        history.append(rec)
        print(f"[Epoch {epoch+1}/{cfg.epochs_ssl}] Loss promedio: {rec['avg_loss']:.4f} | Tiempo: {rec['time_sec']:.1f}s")

    checkpoint_path = os.path.join(cfg.checkpoint_dir, f"{method}_{data_config}_{cfg.mode}_final.pt")
    torch.save({
        "method": method,
        "data_config": data_config,
        "backbone_state_dict": get_backbone_state_dict(model, method),
        "full_model_state_dict": model.state_dict(),
        "history": history,
        "cfg_mode": cfg.mode,
    }, checkpoint_path)

    history_path = os.path.join(cfg.results_dir, f"history_{method}_{data_config}_{cfg.mode}.npy")
    np.save(history_path, np.array(history, dtype=object))
    print(f"\nCheckpoint guardado en: {checkpoint_path}")
    print(f"Historial guardado en: {history_path}")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["debug", "full"], default="debug")
    parser.add_argument("--method", choices=["simsiam", "byol", "cpc", "align_uniform"], required=True)
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--use_dummy", action="store_true")
    args = parser.parse_args()
    cfg = get_config(args.mode)
    print(f"Configuracion: {cfg}")
    train(cfg, args.method, args.data_config, args.use_dummy)


if __name__ == "__main__":
    main()
