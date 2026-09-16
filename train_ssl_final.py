"""train_ssl_final.py
Entrenamiento final controlado usando config_final.py.

Uso:
  python train_ssl_final.py --method simsiam --data_config full
  python train_ssl_final.py --method byol --data_config sas_keep_40pct
  python train_ssl_final.py --method cpc --data_config random_keep_60pct

Los resultados se guardan en checkpoints_final/ y results_final/.
"""

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config_final import get_config
from data_utils import load_food101, get_ssl_augmentation, TwoViewDataset
from ssl_models import build_ssl_model
from ssl_losses import simsiam_loss, byol_loss, cpc_loss, align_uniform_loss


class CPCDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform, n_rows=4):
        self.base_dataset = base_dataset
        self.transform = transform
        self.n_rows = n_rows

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        tensor = self.transform(image)
        _, h, _ = tensor.shape
        if h % self.n_rows != 0:
            raise ValueError(f"Altura {h} no divisible entre {self.n_rows}")
        step = h // self.n_rows
        strips = torch.stack([
            tensor[:, i * step:(i + 1) * step, :]
            for i in range(self.n_rows)
        ])
        return strips, label, idx


def load_indices(cfg, data_config):
    if data_config == "full":
        return None

    # Candidatos de ruta segun la configuracion y el modo
    candidates = [
        os.path.join(cfg.subset_dir, f"{data_config}_{cfg.mode}.npy"),
        os.path.join(cfg.subset_dir, f"{data_config}_full.npy"),
        os.path.join(cfg.subset_dir, f"{data_config}.npy"),
    ]

    for path in candidates:
        if os.path.exists(path):
            print(f"Cargando subconjunto desde: {path}")
            return np.load(path).astype(np.int64)

    raise FileNotFoundError(
        f"No se encontro el subconjunto para '{data_config}' en {cfg.subset_dir}.\n"
        f"Rutas intentadas:\n  - " + "\n  - ".join(candidates)
    )


def get_backbone_state_dict(model, method):
    if method in ("simsiam", "align_uniform"):
        return model.backbone.state_dict()
    if method == "byol":
        return model.online_backbone.state_dict()
    if method == "cpc":
        return model.encoder.backbone.state_dict()
    raise ValueError(f"Metodo no valido: {method}")


def training_step(method, model, batch, device):
    nb = device.type == "cuda"
    if method == "simsiam":
        x1, x2, _, _ = batch
        p1, z2, p2, z1 = model(x1.to(device, non_blocking=nb), x2.to(device, non_blocking=nb))
        return simsiam_loss(p1, z2, p2, z1), {}
    if method == "byol":
        x1, x2, _, _ = batch
        q1, tz2, q2, tz1 = model(x1.to(device, non_blocking=nb), x2.to(device, non_blocking=nb))
        return byol_loss(q1, tz2, q2, tz1), {}
    if method == "align_uniform":
        x1, x2, _, _ = batch
        z1, z2 = model(x1.to(device, non_blocking=nb), x2.to(device, non_blocking=nb))
        loss, la, lu = align_uniform_loss(z1, z2)
        return loss, {"l_align": la.item(), "l_uniform": lu.item()}
    if method == "cpc":
        strips, _, _ = batch
        z, _, predictions = model(strips.to(device, non_blocking=nb))
        return cpc_loss(predictions, z), {}
    raise ValueError(f"Metodo no valido: {method}")


def train(method, data_config, mode="full", use_dummy=False):
    cfg = get_config(mode)
    print(f"Configuracion: {cfg}")
    print(f"Metodo: {method} | Datos: {data_config} | Modo: {mode} | Device: {cfg.device}")

    if use_dummy:
        from data_utils import generar_dataset_dummy
        dataset = generar_dataset_dummy(cfg)
    else:
        dataset = load_food101(cfg, split="train")

    indices = load_indices(cfg, data_config)
    base = Subset(dataset, indices.tolist()) if indices is not None else dataset
    n_used = len(base)
    print(f"Dataset: {len(dataset)} | Muestras utilizadas: {n_used}")

    transform = get_ssl_augmentation(cfg.image_size)
    wrapped = CPCDataset(base, transform) if method == "cpc" else TwoViewDataset(base, transform)
    loader = DataLoader(
        wrapped,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        drop_last=True,
    )

    model = build_ssl_model(method, cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)
    history = []

    for epoch in range(cfg.epochs_ssl):
        start = time.perf_counter()
        model.train()
        losses = []
        logs_acc = {}

        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
                loss, logs = training_step(method, model, batch, cfg.device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if method == "byol":
                model.update_target_network()

            losses.append(float(loss.detach().cpu()))
            for key, value in logs.items():
                logs_acc.setdefault(key, []).append(value)

            if batch_idx % cfg.log_every == 0:
                print(f"Epoch {epoch+1}/{cfg.epochs_ssl} | Batch {batch_idx+1}/{len(loader)} | Loss {loss.item():.4f}")

        record = {
            "epoch": epoch + 1,
            "avg_loss": float(np.mean(losses)),
            "time_sec": time.perf_counter() - start,
        }
        for key, values in logs_acc.items():
            record[f"avg_{key}"] = float(np.mean(values))
        history.append(record)
        print(f"[Epoch {epoch+1}/{cfg.epochs_ssl}] loss={record['avg_loss']:.4f} time={record['time_sec']:.1f}s")

    checkpoint_path = os.path.join(cfg.checkpoint_dir, f"{method}_{data_config}_final.pt")
    torch.save({
        "method": method,
        "data_config": data_config,
        "backbone_state_dict": get_backbone_state_dict(model, method),
        "full_model_state_dict": model.state_dict(),
        "history": history,
        "cfg_mode": cfg.mode,
    }, checkpoint_path)

    history_path = os.path.join(cfg.results_dir, f"history_{method}_{data_config}.npy")
    np.save(history_path, np.array(history, dtype=object))
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Historial: {history_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["simsiam", "byol", "cpc", "align_uniform"], required=True)
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--mode", choices=["debug", "full"], default="full")
    parser.add_argument("--use_dummy", action="store_true")
    args = parser.parse_args()
    train(args.method, args.data_config, mode=args.mode, use_dummy=args.use_dummy)


if __name__ == "__main__":
    main()
