"""
train_ssl.py
Script de entrenamiento UNIFICADO para los 4 metodos SSL (SimSiam, BYOL, CPC,
Align-Uniform), sobre CUALQUIER configuracion de datos (dataset completo, o
uno de los subconjuntos SAS/aleatorios generados por generate_subsets.py).

TERMINOS CLAVE (recordatorio):
- "Checkpoint": archivo que guarda el estado de un modelo entrenado (sus
  pesos/parametros) en un momento dado, para poder retomar el entrenamiento
  o usarlo despues sin tener que re-entrenar desde cero.
- "Epoca" (epoch): una pasada completa por TODOS los datos de entrenamiento.
- "Optimizador": el algoritmo que ajusta los pesos de la red en cada paso,
  usando el gradiente de la perdida (aqui usamos Adam, ver config.py).

FLUJO DE DATOS SEGUN EL METODO:
- SimSiam, BYOL, Align-Uniform: usan el mismo formato de datos (dos vistas
  aumentadas completas de cada imagen, via data_utils.TwoViewDataset).
- CPC: usa un formato DISTINTO (franjas horizontales de una sola vista de
  cada imagen, ver funcion image_to_strips en este archivo), porque su
  arquitectura es secuencial/autoregresiva, no basada en pares de vistas.

CORRECCION IMPORTANTE (encontrada durante revision antes de la entrega):
Cada una de las 4 clases en ssl_models.py guarda su backbone en un atributo
DISTINTO: SimSiam y AlignUniformModel usan 'model.backbone', BYOL usa
'model.online_backbone', y CPC lo anida en 'model.encoder.backbone'. La
version anterior de este archivo solo cubria los primeros dos casos, lo que
habria causado un AttributeError AL FINAL del entrenamiento de CPC (es decir,
DESPUES de invertir todo el tiempo de computo, justo al intentar guardar el
checkpoint). Se agrego la funcion get_backbone_state_dict() que maneja
explicitamente los 3 casos posibles.

USO:
    python train_ssl.py --mode debug --method simsiam --data_config full --use_dummy
    python train_ssl.py --mode debug --method byol --data_config sas_20pct --use_dummy
    python train_ssl.py --mode full --method cpc --data_config random_60pct

    (En la PC de la universidad, sin --use_dummy, se usa Food-101 real)
"""

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import get_config
from data_utils import (
    load_food101, generar_dataset_dummy, get_ssl_augmentation,
    TwoViewDataset, SingleViewDataset
)
from ssl_models import build_ssl_model
from ssl_losses import simsiam_loss, byol_loss, cpc_loss, align_uniform_loss


# ----------------------------------------------------------------------------
# 1. Utilidades de datos especificas para CPC (particion en franjas)
# ----------------------------------------------------------------------------

def image_to_strips(image_tensor, n_rows=4):
    """
    Divide una imagen (C, H, W) en n_rows franjas horizontales, para usarlas
    como secuencia de entrada al modelo CPC (ver ssl_models.CPC, que espera
    (batch, n_rows, C, strip_H, W)).

    Esta logica ya fue validada por separado con numpy antes de escribir
    esta version en PyTorch (ver conversacion del proyecto).
    """
    C, H, W = image_tensor.shape
    strip_height = H // n_rows
    strips = [image_tensor[:, i*strip_height:(i+1)*strip_height, :] for i in range(n_rows)]
    return torch.stack(strips, dim=0)  # (n_rows, C, strip_height, W)


class CPCDataset(torch.utils.data.Dataset):
    """
    Wrapper de dataset que devuelve una SOLA vista de cada imagen, ya dividida
    en franjas horizontales, en vez de dos vistas completas (formato requerido
    por SimSiam/BYOL/Align-Uniform).
    """
    def __init__(self, base_dataset, transform, n_rows=4):
        self.base_dataset = base_dataset
        self.transform = transform
        self.n_rows = n_rows

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        img_tensor = self.transform(img)
        strips = image_to_strips(img_tensor, n_rows=self.n_rows)
        return strips, label, idx


# ----------------------------------------------------------------------------
# 2. Carga de datos segun la configuracion pedida (full / sas_Xpct / random_Xpct)
# ----------------------------------------------------------------------------

def load_dataset_and_indices(cfg, data_config, use_dummy):
    """
    Args:
        data_config: "full", "sas_10pct", "sas_20pct", "sas_40pct", "sas_60pct",
                     "random_10pct", "random_20pct", "random_40pct", "random_60pct"
        use_dummy: si usar el dataset dummy en vez de Food-101 real

    Returns:
        (dataset, indices): el dataset base y la lista de indices a usar
        (si data_config == "full", indices es None y se usan TODOS)
    """
    if use_dummy:
        dataset = generar_dataset_dummy(cfg)
    else:
        dataset = load_food101(cfg, split="train")

    if data_config == "full":
        return dataset, None

    subset_path = os.path.join(cfg.subset_dir, f"{data_config}_{cfg.mode}.npy")
    if not os.path.exists(subset_path):
        raise FileNotFoundError(
            f"No se encontro el archivo de subconjunto '{subset_path}'. "
            f"Debes correr generate_subsets.py primero con el mismo --mode "
            f"y --use_dummy para generar los indices de '{data_config}'."
        )
    indices = np.load(subset_path)
    return dataset, indices


# ----------------------------------------------------------------------------
# 3. Extraccion del backbone para guardado de checkpoint (CORREGIDO: cubre
#    los 3 casos distintos de estructura interna de cada clase en ssl_models.py)
# ----------------------------------------------------------------------------

def get_backbone_state_dict(model, method):
    """
    Cada clase en ssl_models.py guarda su backbone en un atributo distinto:
      - SimSiam, AlignUniformModel: model.backbone
      - BYOL: model.online_backbone (la red que SI se entrena por gradiente)
      - CPC: model.encoder.backbone (anidado dentro de PatchEncoder)

    Esta funcion centraliza esa logica para evitar errores de atributo al
    guardar checkpoints (bug detectado y corregido antes de la entrega).
    """
    if method in ("simsiam", "align_uniform"):
        return model.backbone.state_dict()
    elif method == "byol":
        return model.online_backbone.state_dict()
    elif method == "cpc":
        return model.encoder.backbone.state_dict()
    else:
        raise ValueError(f"Metodo '{method}' no reconocido en get_backbone_state_dict")


# ----------------------------------------------------------------------------
# 4. Un paso de entrenamiento por metodo (forward + loss)
# ----------------------------------------------------------------------------

def training_step(method, model, batch, device):
    """
    Ejecuta el forward pass y calcula la perdida para UN batch, segun el
    metodo SSL. Devuelve (loss, logs_dict) donde logs_dict contiene metricas
    adicionales utiles para monitorear (ej. L_align y L_uniform por separado).
    """
    logs = {}

    if method == "simsiam":
        x1, x2, _, _ = batch
        x1, x2 = x1.to(device), x2.to(device)
        p1, z2, p2, z1 = model(x1, x2)
        loss = simsiam_loss(p1, z2, p2, z1)

    elif method == "byol":
        x1, x2, _, _ = batch
        x1, x2 = x1.to(device), x2.to(device)
        q1, target_z2, q2, target_z1 = model(x1, x2)
        loss = byol_loss(q1, target_z2, q2, target_z1)

    elif method == "align_uniform":
        x1, x2, _, _ = batch
        x1, x2 = x1.to(device), x2.to(device)
        z1, z2 = model(x1, x2)
        loss, l_align, l_uniform = align_uniform_loss(z1, z2)
        logs["l_align"] = l_align.item()
        logs["l_uniform"] = l_uniform.item()

    elif method == "cpc":
        strips, _, _ = batch
        strips = strips.to(device)
        z, c, predictions = model(strips)
        loss = cpc_loss(predictions, z)

    else:
        raise ValueError(f"Metodo '{method}' no reconocido")

    return loss, logs


# ----------------------------------------------------------------------------
# 5. Loop principal de entrenamiento
# ----------------------------------------------------------------------------

def train(cfg, method, data_config, use_dummy=False):
    print(f"\n{'='*70}")
    print(f"Entrenando: metodo={method} | datos={data_config} | modo={cfg.mode}")
    print(f"{'='*70}\n")

    dataset, indices = load_dataset_and_indices(cfg, data_config, use_dummy)
    n_used = len(indices) if indices is not None else len(dataset)
    print(f"Dataset base: {len(dataset)} muestras | Usando: {n_used} muestras ({data_config})")

    eval_augmentation = get_ssl_augmentation(cfg.image_size)

    base = Subset(dataset, indices.tolist()) if indices is not None else dataset

    if method == "cpc":
        wrapped = CPCDataset(base, eval_augmentation, n_rows=4)
    else:
        wrapped = TwoViewDataset(base, eval_augmentation)

    loader = DataLoader(wrapped, batch_size=cfg.batch_size, shuffle=True,
                         num_workers=cfg.num_workers, drop_last=True)

    if len(loader) == 0:
        raise ValueError(
            f"El DataLoader quedo vacio (0 batches). Esto pasa si "
            f"n_used={n_used} < cfg.batch_size={cfg.batch_size} (drop_last=True "
            f"descarta el ultimo batch incompleto). Reduce cfg.batch_size en "
            f"config.py o usa un subconjunto/dataset mas grande."
        )

    model = build_ssl_model(method, cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                   weight_decay=cfg.weight_decay)

    history = []

    for epoch in range(cfg.epochs_ssl):
        epoch_start = time.time()
        model.train()
        epoch_losses = []
        epoch_logs = {}

        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()
            loss, logs = training_step(method, model, batch, cfg.device)
            loss.backward()
            optimizer.step()

            if method == "byol":
                model.update_target_network()

            epoch_losses.append(loss.item())
            for k, v in logs.items():
                epoch_logs.setdefault(k, []).append(v)

            if batch_idx % cfg.log_every == 0:
                print(f"  Epoch {epoch+1}/{cfg.epochs_ssl} | Batch {batch_idx+1}/{len(loader)} "
                      f"| Loss: {loss.item():.4f}")

        avg_loss = float(np.mean(epoch_losses))
        epoch_time = time.time() - epoch_start
        record = {"epoch": epoch + 1, "avg_loss": avg_loss, "time_sec": epoch_time}
        for k, v in epoch_logs.items():
            record[f"avg_{k}"] = float(np.mean(v))
        history.append(record)

        print(f"[Epoch {epoch+1}/{cfg.epochs_ssl}] Loss promedio: {avg_loss:.4f} "
              f"| Tiempo: {epoch_time:.1f}s")

    # --- Guardado del checkpoint final (usa la funcion corregida) ---
    checkpoint_name = f"{method}_{data_config}_{cfg.mode}_final.pt"
    checkpoint_path = os.path.join(cfg.checkpoint_dir, checkpoint_name)

    checkpoint_dict = {
        "method": method,
        "data_config": data_config,
        "backbone_state_dict": get_backbone_state_dict(model, method),
        "full_model_state_dict": model.state_dict(),
        "history": history,
        "cfg_mode": cfg.mode,
    }
    torch.save(checkpoint_dict, checkpoint_path)
    print(f"\nCheckpoint guardado en: {checkpoint_path}")

    # --- Guardado del historial de entrenamiento (para graficar despues) ---
    history_path = os.path.join(cfg.results_dir, f"history_{method}_{data_config}_{cfg.mode}.npy")
    np.save(history_path, history)
    print(f"Historial guardado en: {history_path}")

    return history


# ----------------------------------------------------------------------------
# 6. Punto de entrada (linea de comandos)
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo SSL sobre una configuracion de datos")
    parser.add_argument("--mode", type=str, default="debug", choices=["debug", "full"])
    parser.add_argument("--method", type=str, required=True,
                         choices=["simsiam", "byol", "cpc", "align_uniform"])
    parser.add_argument("--data_config", type=str, required=True,
                         help="'full', 'sas_10pct', 'sas_20pct', 'sas_40pct', 'sas_60pct', "
                              "'random_10pct', 'random_20pct', 'random_40pct', 'random_60pct'")
    parser.add_argument("--use_dummy", action="store_true")
    args = parser.parse_args()

    cfg = get_config(args.mode)
    print(f"Configuracion: {cfg}")

    train(cfg, args.method, args.data_config, use_dummy=args.use_dummy)


if __name__ == "__main__":
    main()
