"""check_progress.py
Muestra el estado en tiempo real de los experimentos y checkpoints generados.
Uso:
  python check_progress.py
"""

import os
from config_final import get_config

METHODS = ["simsiam", "byol", "cpc", "align_uniform"]
CONFIGS = ["full", "sas_keep_80pct", "sas_keep_60pct", "random_keep_60pct"]

MACHINE_MAP = {
    "simsiam": "Kaggle",
    "byol": "Kaggle",
    "cpc": "PC U (RTX A2000)",
    "align_uniform": "PC U (RTX A2000)",
}


def main():
    cfg = get_config("full")
    ckpt_dir = cfg.checkpoint_dir

    print("=" * 80)
    print(f"ESTADO DE EXPERIMENTOS SSL - PAE")
    print(f"Directorio de checkpoints: {os.path.abspath(ckpt_dir)}")
    print("=" * 80)

    header = f"{'#':<3} | {'Metodo':<14} | {'Data Config':<20} | {'Maquina':<18} | {'Estado':<10} | {'Tamano'}"
    print(header)
    print("-" * 80)

    total = len(METHODS) * len(CONFIGS)
    completed = 0
    idx = 0

    for method in METHODS:
        for data_config in CONFIGS:
            idx += 1
            ckpt_name = f"{method}_{data_config}_final.pt"
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            machine = MACHINE_MAP.get(method, "Cualquiera")

            if os.path.exists(ckpt_path):
                size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
                status = "[OK] LISTO"
                size_str = f"{size_mb:.1f} MB"
                completed += 1
            else:
                status = "[..] Pendiente"
                size_str = "-"

            print(f"{idx:<3} | {method:<14} | {data_config:<20} | {machine:<18} | {status:<10} | {size_str}")

    pct = (completed / total) * 100
    print("-" * 80)
    print(f"Progreso global: {completed}/{total} experimentos listos ({pct:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
