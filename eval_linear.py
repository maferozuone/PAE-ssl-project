"""eval_linear.py
Protocolo estandar de Evaluacion Lineal (Linear Probing) para modelos SSL preentrenados.

Evalua la calidad de las representaciones del backbone congelando sus pesos y
entrenando una cabeza de clasificacion lineal sobre Food-101.

Uso:
  python eval_linear.py --method simsiam --data_config full --mode full
  python eval_linear.py --checkpoint output/checkpoints_final/simsiam_full_final.pt
  python eval_linear.py --all --mode full
  python eval_linear.py --mode debug --use_dummy
"""

import argparse
import glob
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config_final import get_config
from data_utils import (
    load_food101,
    generar_dataset_dummy,
    get_linear_train_transform,
    get_eval_transform,
    SingleViewDataset,
)
from ssl_models import build_shared_backbone


class LinearClassifier(nn.Module):
    """Backbone congelado + cabeza de clasificacion lineal."""
    def __init__(self, backbone, embedding_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.fc = nn.Linear(embedding_dim, num_classes)
        # Inicializacion normal con std=0.01 y bias=0
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.fc(features)


@torch.no_grad()
def accuracy(output, target, topk=(1, 5)):
    """Calcula la precision Top-k para los k especificados."""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(float(correct_k.mul_(100.0 / batch_size).item()))
    return res


def load_backbone_from_checkpoint(checkpoint_path, backbone_name="resnet18", device="cpu"):
    """Carga los pesos del backbone desde un checkpoint SSL."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone, embedding_dim = build_shared_backbone(backbone_name, pretrained=False)

    if "backbone_state_dict" in checkpoint:
        state_dict = checkpoint["backbone_state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Limpiar prefijos si los hubiera (ej. 'backbone.', 'online_backbone.', 'module.')
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_k = k
        for prefix in ["backbone.", "online_backbone.", "encoder.backbone.", "module."]:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix):]
        cleaned_state_dict[clean_k] = v

    missing, unexpected = backbone.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"  [Aviso] Claves faltantes al cargar backbone: {len(missing)}")
    if unexpected:
        print(f"  [Aviso] Claves inesperadas ignoradas: {len(unexpected)}")

    backbone.eval()
    return backbone, embedding_dim, checkpoint


def evaluate(model, loader, criterion, device, use_amp=True):
    """Evalua el modelo sobre el split de prueba calculando Top-1 y Top-5."""
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    nb = device.type == "cuda"
    with torch.no_grad():
        for imgs, targets, _ in loader:
            imgs = imgs.to(device, non_blocking=nb)
            targets = targets.to(device, non_blocking=nb)
            batch_size = targets.size(0)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            total_loss += loss.item() * batch_size
            total_top1 += prec1 * batch_size
            total_top5 += prec5 * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_top1 = total_top1 / total_samples
    avg_top5 = total_top5 / total_samples
    return avg_loss, avg_top1, avg_top5


def train_linear_eval(cfg, checkpoint_path, epochs=None, lr=0.1, optimizer_name="sgd",
                      use_dummy=False):
    """Ejecuta el protocolo de evaluacion lineal para un checkpoint dado."""
    epochs = epochs or cfg.epochs_linear_eval
    device = cfg.device
    print(f"\n{'='*75}")
    print(f"EVALUACION LINEAL: {os.path.basename(checkpoint_path)}")
    print(f"Dispositivo: {device} | Epocas: {epochs} | Batch size: {cfg.batch_size} | Opt: {optimizer_name} (lr={lr})")
    print(f"{'='*75}")

    # 1. Carga de datasets
    if use_dummy:
        print("Usando datasets DUMMY para validacion...")
        train_dataset = generar_dataset_dummy(cfg)
        test_dataset = generar_dataset_dummy(cfg)
    else:
        print("Cargando Food-101 train y test...")
        train_dataset = load_food101(cfg, split="train")
        test_dataset = load_food101(cfg, split="test")

    train_transform = get_linear_train_transform(cfg.image_size)
    test_transform = get_eval_transform(cfg.image_size)

    train_loader = DataLoader(
        SingleViewDataset(train_dataset, train_transform),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg.num_workers > 0),
        drop_last=False,
    )
    test_loader = DataLoader(
        SingleViewDataset(test_dataset, test_transform),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(cfg.num_workers > 0),
        drop_last=False,
    )
    print(f"Muestras de entrenamiento: {len(train_dataset)} | Muestras de prueba: {len(test_dataset)}")

    # 2. Cargar Backbone y construir LinearClassifier
    backbone, emb_dim, ckpt_meta = load_backbone_from_checkpoint(
        checkpoint_path, backbone_name=cfg.backbone, device=device
    )
    model = LinearClassifier(backbone, embedding_dim=emb_dim, num_classes=cfg.num_classes).to(device)

    # 3. Configurar optimizador solo para la capa lineal
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    elif optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.fc.parameters(), lr=lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Optimizador no soportado: {optimizer_name}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp)

    method = ckpt_meta.get("method", "unknown") if isinstance(ckpt_meta, dict) else "unknown"
    data_config = ckpt_meta.get("data_config", "unknown") if isinstance(ckpt_meta, dict) else "unknown"

    history = []
    best_top1 = 0.0
    best_top5 = 0.0
    best_epoch = 0

    nb = device.type == "cuda"

    # 4. Bucle de evaluacion
    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        model.fc.train()
        train_loss_accum = 0.0
        train_top1_accum = 0.0
        train_total = 0

        for batch_idx, (imgs, targets, _) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=nb)
            targets = targets.to(device, non_blocking=nb)
            batch_size = targets.size(0)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            prec1, _ = accuracy(outputs, targets, topk=(1, 5))
            train_loss_accum += loss.item() * batch_size
            train_top1_accum += prec1 * batch_size
            train_total += batch_size

        scheduler.step()
        train_loss = train_loss_accum / train_total
        train_top1 = train_top1_accum / train_total

        # Evaluacion en test
        test_loss, test_top1, test_top5 = evaluate(
            model, test_loader, criterion, device, use_amp=cfg.use_amp
        )
        elapsed = time.perf_counter() - start_time

        if test_top1 > best_top1:
            best_top1 = test_top1
            best_top5 = test_top5
            best_epoch = epoch

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "test_loss": test_loss,
            "test_top1": test_top1,
            "test_top5": test_top5,
            "lr": scheduler.get_last_lr()[0],
            "time_sec": elapsed,
        }
        history.append(record)

        print(
            f"[Eval Epoca {epoch:03d}/{epochs:03d}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_top1:.2f}% | "
            f"Test Loss: {test_loss:.4f} Top-1: {test_top1:.2f}% Top-5: {test_top5:.2f}% | "
            f"({elapsed:.1f}s)"
        )

    print(f"\n--> MEJOR RESULTADO: Top-1: {best_top1:.2f}% | Top-5: {best_top5:.2f}% (Epoca {best_epoch})")

    # 5. Guardar resultados
    os.makedirs(cfg.results_dir, exist_ok=True)
    base_name = f"eval_{method}_{data_config}"
    json_path = os.path.join(cfg.results_dir, f"{base_name}.json")
    npy_path = os.path.join(cfg.results_dir, f"{base_name}_history.npy")

    summary = {
        "checkpoint": checkpoint_path,
        "method": method,
        "data_config": data_config,
        "mode": cfg.mode,
        "epochs": epochs,
        "best_top1": best_top1,
        "best_top5": best_top5,
        "best_epoch": best_epoch,
        "final_top1": history[-1]["test_top1"] if history else 0.0,
        "final_top5": history[-1]["test_top5"] if history else 0.0,
        "history": history,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    np.save(npy_path, np.array(history, dtype=object))
    print(f"Resultados guardados en:\n  {json_path}\n  {npy_path}\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluacion lineal para modelos SSL en Food-101")
    parser.add_argument("--mode", choices=["debug", "full"], default="full")
    parser.add_argument("--checkpoint", type=str, default=None, help="Ruta al checkpoint .pt")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directorio de checkpoints")
    parser.add_argument("--method", choices=["simsiam", "byol", "cpc", "align_uniform"], default=None)
    parser.add_argument("--data_config", type=str, default=None)
    parser.add_argument("--all", action="store_true", help="Evaluar todos los checkpoints en checkpoint_dir")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--use_dummy", action="store_true", help="Usar dataset dummy para pruebas rapidas")
    args = parser.parse_args()

    cfg = get_config(args.mode)
    ckpt_dir = args.checkpoint_dir or cfg.checkpoint_dir
    # Si ckpt_dir no tiene archivos .pt, intentar con output/checkpoints
    if not glob.glob(os.path.join(ckpt_dir, "*.pt")):
        alt_dir = os.path.join(cfg.output_root, "checkpoints")
        if glob.glob(os.path.join(alt_dir, "*.pt")):
            ckpt_dir = alt_dir

    if args.all:
        pattern = os.path.join(ckpt_dir, "*.pt")
        checkpoints = sorted(glob.glob(pattern))
        if not checkpoints:
            print(f"No se encontraron checkpoints en {ckpt_dir}")
            return
        print(f"Se encontraron {len(checkpoints)} checkpoints para evaluar en {ckpt_dir}:")
        for ckpt in checkpoints:
            print(f" - {ckpt}")
        for ckpt in checkpoints:
            train_linear_eval(
                cfg, ckpt, epochs=args.epochs, lr=args.lr,
                optimizer_name=args.optimizer, use_dummy=args.use_dummy
            )
        return

    # Determinacion del checkpoint a evaluar
    if args.checkpoint:
        ckpt_path = args.checkpoint
    elif args.method and args.data_config:
        ckpt_path = os.path.join(ckpt_dir, f"{args.method}_{args.data_config}_final.pt")
        if not os.path.exists(ckpt_path):
            # intentar con patron comodin
            candidates = glob.glob(os.path.join(ckpt_dir, f"{args.method}_{args.data_config}*.pt"))
            if candidates:
                ckpt_path = candidates[0]
            else:
                raise FileNotFoundError(f"No se encontro checkpoint para {args.method} {args.data_config} en {ckpt_dir}")
    else:
        pattern = os.path.join(ckpt_dir, "*.pt")
        available = glob.glob(pattern)
        if available:
            ckpt_path = available[0]
            print(f"No se especifico checkpoint. Usando el primero encontrado: {ckpt_path}")
        else:
            raise ValueError(f"Debes especificar --checkpoint o (--method y --data_config). No hay .pt en {ckpt_dir}")

    train_linear_eval(
        cfg, ckpt_path, epochs=args.epochs, lr=args.lr,
        optimizer_name=args.optimizer, use_dummy=args.use_dummy
    )


if __name__ == "__main__":
    main()
