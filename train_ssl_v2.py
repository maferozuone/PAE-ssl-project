"""
train_ssl_v2.py
Entrenamiento SSL compatible con los nuevos subconjuntos:

    sas_keep_90pct_full.npy
    sas_keep_80pct_full.npy
    sas_keep_60pct_full.npy
    sas_keep_40pct_full.npy
    random_keep_90pct_full.npy
    random_keep_80pct_full.npy
    random_keep_60pct_full.npy
    random_keep_40pct_full.npy

La palabra keep indica la fraccion CONSERVADA. Por ejemplo:
    sas_keep_40pct = subconjunto SAS que conserva 40% y elimina 60%.

Este archivo importa config_v2.py para usar la configuracion actualizada.
No reemplaza automaticamente el train_ssl.py anterior.

USO:
    python train_ssl_v2.py --mode debug --method simsiam --data_config full --use_dummy
    python train_ssl_v2.py --mode full --method simsiam --data_config sas_keep_40pct
    python train_ssl_v2.py --mode full --method byol --data_config random_keep_60pct
    python train_ssl_v2.py --mode full --method cpc --data_config sas_keep_80pct
"""

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config_v2 import get_config
from data_utils import (
    load_food101, generar_dataset_dummy, get_ssl_augmentation,
    TwoViewDataset
)
from ssl_models import build_ssl_model
from ssl_losses import simsiam_loss, byol_loss, cpc_loss, align_uniform_loss


# ----------------------------------------------------------------------------
# 1. Datos para CPC
# ----------------------------------------------------------------------------

def image_to_strips(image_tensor, n_rows=4):
    """Divide una imagen (C,H,W) en franjas horizontales."""
    C, H, W = image_tensor.shape
    if H % n_rows != 0:
        raise ValueError(f"La altura {H} no es divisible entre n_rows={n_rows}")
    strip_height = H // n_rows
    return torch.stack([
        image_tensor[:, i * strip_height:(i + 1) * strip_height, :]
        for i in range(n_rows)
    ], dim=0)


class CPCDataset(torch.utils.data.Dataset):
    """Dataset de una vista transformada dividida en franjas para CPC."""
    def __init__(self, base_dataset, transform, n_rows=4):
        self.base_dataset = base_dataset
        self.transform = transform
        self.n_rows = n_rows

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        image_tensor = self.transform(image)
        strips = image_to_strips(image_tensor, self.n_rows)
        return strips, label, idx


# ----------------------------------------------------------------------------
# 2. Carga de configuraciones de datos
# ----------------------------------------------------------------------------

def valid_data_configs():
    return {
        "full",
        "sas_keep_90pct", "sas_keep_80pct", "sas_keep_60pct", "sas_keep_40pct",
        "random_keep_90pct", "random_keep_80pct", "random_keep_60pct", "random_keep_40pct",
    }


def load_dataset_and_indices(cfg, data_config, use_dummy=False):
    """
    Carga Food-101 o el dummy y recupera los indices del subconjunto solicitado.

    Para 'full' devuelve indices=None.
    Para subconjuntos reales busca archivos con sufijo '_full', porque fueron
    generados mediante generate_subsets_fast.py --mode full.
    """
    if data_config not in valid_data_configs():
        raise ValueError(
            f"data_config='{data_config}' no valido. Opciones: "
            f"{sorted(valid_data_configs())}"
        )

    if use_dummy:
        dataset = generar_dataset_dummy(cfg)
    else:
        dataset = load_food101(cfg, split="train")

    if data_config == "full":
        return dataset, None

    # En modo debug los indices dummy tienen sufijo debug; en modo full,
    # los indices Food-101 tienen sufijo full.
    suffix = "debug" if use_dummy and cfg.mode == "debug" else "full"
    subset_path = os.path.join(cfg.subset_dir, f"{data_config}_{suffix}.npy")

    if not os.path.exists(subset_path):
        raise FileNotFoundError(
            f"No se encontro '{subset_path}'. Ejecuta primero:\n"
            f"  python generate_subsets_fast.py --mode {cfg.mode}"
            f"{' --use_dummy' if use_dummy else ''}\n"
            f"o verifica que el archivo de indices tenga el nombre exacto."
        )

    indices = np.load(subset_path).astype(np.int64)

    # Validacion para detectar incompatibilidad entre indices y dataset.
    if len(indices) == 0:
        raise ValueError(f"El archivo de indices esta vacio: {subset_path}")
    if np.min(indices) < 0 or np.max(indices) >= len(dataset):
        raise IndexError(
            f"Los indices de '{subset_path}' no corresponden al dataset cargado. "
            f"Rango de indices: [{np.min(indices)}, {np.max(indices)}], "
            f"tamano del dataset: {len(dataset)}."
        )

    return dataset, indices


# ----------------------------------------------------------------------------
# 3. Backbone para checkpoint
# ----------------------------------------------------------------------------

def get_backbone_state_dict(model, method):
    if method in ("simsiam", "align_uniform"):
        return model.backbone.state_dict()
    if method == "byol":
        return model.online_backbone.state_dict()
    if method == "cpc":
        return model.encoder.backbone.state_dict()
    raise ValueError(f"Metodo '{method}' no reconocido")


# ----------------------------------------------------------------------------
# 4. Paso de entrenamiento
# ----------------------------------------------------------------------------

def training_step(method, model, batch, device):
    logs = {}

    if method == "simsiam":
        x1, x2, _, _ = batch
        x1 = x1.to(device, non_blocking=(device.type == "cuda"))
        x2 = x2.to(device, non_blocking=(device.type == "cuda"))
        p1, z2, p2, z1 = model(x1, x2)
        loss = simsiam_loss(p1, z2, p2, z1)

    elif method == "byol":
        x1, x2, _, _ = batch
        x1 = x1.to(device, non_blocking=(device.type == "cuda"))
        x2 = x2.to(device, non_blocking=(device.type == "cuda"))
        q1, target_z2, q2, target_z1 = model(x1, x2)
        loss = byol_loss(q1, target_z2, q2, target_z1)

    elif method == "align_uniform":
        x1, x2, _, _ = batch
        x1 = x1.to(device, non_blocking=(device.type == "cuda"))
        x2 = x2.to(device, non_blocking=(device.type == "cuda"))
        z1, z2 = model(x1, x2)
        loss, l_align, l_uniform = align_uniform_loss(z1, z2)
        logs["l_align"] = l_align.item()
        logs["l_uniform"] = l_uniform.item()

    elif method == "cpc":
        strips, _, _ = batch
        strips = strips.to(device, non_blocking=(device.type == "cuda"))
        z, _, predictions = model(strips)
        loss = cpc_loss(predictions, z)

    else:
        raise ValueError(f"Metodo '{method}' no reconocido")

    return loss, logs


# ----------------------------------------------------------------------------
# 5. Entrenamiento
# ----------------------------------------------------------------------------

def train(cfg, method, data_config, use_dummy=False):
    print(f"\n{'=' * 70}")
    print(f"Entrenando: metodo={method} | datos={data_config} | modo={cfg.mode}")
    print(f"{'=' * 70}\n")

    dataset, indices = load_dataset_and_indices(cfg, data_config, use_dummy)
    n_used = len(indices) if indices is not None else len(dataset)
    print(f"Dataset base: {len(dataset)} | Usando: {n_used} muestras")
    print(f"Dispositivo: {cfg.device}")

    base = Subset(dataset, indices.tolist()) if indices is not None else dataset
    transform = get_ssl_augmentation(cfg.image_size)

    if method == "cpc":
        wrapped = CPCDataset(base, transform, n_rows=4)
    else:
        wrapped = TwoViewDataset(base, transform)

    loader = DataLoader(
        wrapped,
        batch_size=min(cfg.batch_size, n_used),
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        drop_last=True,
    )

    if len(loader) == 0:
        raise ValueError(
            f"No se pudo formar ningun batch. n_used={n_used}, "
            f"batch_size={cfg.batch_size}."
        )

    model = build_ssl_model(method, cfg).to(cfg.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.device.type == "cuda" and cfg.use_amp))
    history = []

    for epoch in range(cfg.epochs_ssl):
        start = time.perf_counter()
        model.train()
        losses = []
        extra_logs = {}

        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=cfg.device.type,
                dtype=torch.float16,
                enabled=(cfg.device.type == "cuda" and cfg.use_amp),
            ):
                loss, logs = training_step(method, model, batch, cfg.device)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if method == "byol":
                model.update_target_network()

            losses.append(float(loss.detach().cpu().item()))
            for key, value in logs.items():
                extra_logs.setdefault(key, []).append(value)

            if batch_idx % cfg.log_every == 0:
                print(
                    f"  Epoch {epoch + 1}/{cfg.epochs_ssl} | "
                    f"Batch {batch_idx + 1}/{len(loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        record = {
            "epoch": epoch + 1,
            "avg_loss": float(np.mean(losses)),
            "time_sec": time.perf_counter() - start,
        }
        for key, values in extra_logs.items():
            record[f"avg_{key}"] = float(np.mean(values))
        history.append(record)

        print(
            f"[Epoch {epoch + 1}/{cfg.epochs_ssl}] "
            f"Loss promedio: {record['avg_loss']:.4f} | "
            f"Tiempo: {record['time_sec']:.1f}s"
        )

    checkpoint_name = f"{method}_{data_config}_{cfg.mode}_final.pt"
    checkpoint_path = os.path.join(cfg.checkpoint_dir, checkpoint_name)
    torch.save(
        {
            "method": method,
            "data_config": data_config,
            "backbone_state_dict": get_backbone_state_dict(model, method),
            "full_model_state_dict": model.state_dict(),
            "history": history,
            "cfg_mode": cfg.mode,
        },
        checkpoint_path,
    )

    history_path = os.path.join(
        cfg.results_dir,
        f"history_{method}_{data_config}_{cfg.mode}.npy",
    )
    np.save(history_path, np.array(history, dtype=object))

    print(f"\nCheckpoint guardado en: {checkpoint_path}")
    print(f"Historial guardado en: {history_path}")
    return history


# ----------------------------------------------------------------------------
# 6. CLI
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["debug", "full"], default="debug")
    parser.add_argument("--method", choices=["simsiam", "byol", "cpc", "align_uniform"], required=True)
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--use_dummy", action="store_true")
    args = parser.parse_args()

    cfg = get_config(args.mode)
    print(f"Configuracion: {cfg}")
    train(cfg, args.method, args.data_config, use_dummy=args.use_dummy)


if __name__ == "__main__":
    main()
