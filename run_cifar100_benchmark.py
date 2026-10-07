"""run_cifar100_benchmark.py
Script de automatizacion para el benchmark completo o modular de CIFAR-100.

Permite:
1. Validar los 4 modelos SSL (SimSiam, BYOL, CPC, Align-Uniform) en CIFAR-100
   frente a la literatura cientifica (Joshi & Mirzasoleiman ICML 2023, Chen & He 2021).
2. Probar la eficacia de SAS (p. ej. 80% o la matriz completa) en CIFAR-100.
3. Totalmente modular, reanudable e idempotente (se salta lo ya entrenado/evaluado).
4. Completamente aislado: no toca ni sobreescribe ningun resultado de Food-101.

Uso:
  # 1. Solo validacion de modelos en 100% de datos (Full):
  python run_cifar100_benchmark.py --configs full

  # 2. Validacion de modelos + prueba de retencion 80% SAS vs Random:
  python run_cifar100_benchmark.py --configs full sas_keep_80pct random_keep_80pct

  # 3. Matriz completa identica a Food-101 (28 experimentos):
  python run_cifar100_benchmark.py --full_matrix

  # 4. Probar unicamente un metodo (ej. simsiam):
  python run_cifar100_benchmark.py --methods simsiam --configs full
"""

import argparse
import os
import subprocess
import sys
import time
import json
import numpy as np


ALL_METHODS = ["simsiam", "byol", "cpc", "align_uniform"]
ALL_DATA_CONFIGS = [
    "full",
    "sas_keep_90pct", "random_keep_90pct",
    "sas_keep_80pct", "random_keep_80pct",
    "sas_keep_60pct", "random_keep_60pct",
    "sas_keep_40pct", "random_keep_40pct",
]
QUICK_CONFIGS = ["full", "sas_keep_80pct", "random_keep_80pct"]


def run_command(cmd, desc=""):
    print(f"\n{'='*75}")
    print(f"--> [EJECUTANDO] {desc}")
    print(f"Comando: {' '.join(cmd)}")
    print(f"{'='*75}\n")
    start = time.perf_counter()
    res = subprocess.run(cmd)
    elapsed = time.perf_counter() - start
    if res.returncode != 0:
        print(f"[ERROR] Comando fallo con codigo {res.returncode}")
        sys.exit(res.returncode)
    print(f"[COMPLETADO] Tiempo: {elapsed:.1f}s ({elapsed/60:.2f} min)\n")
    return elapsed


def ensure_subsets_exist(configs_needed, mode="full"):
    """Verifica si los subconjuntos requeridos existen en output/subsets_cifar100/. Si no, los genera."""
    non_full_configs = [c for c in configs_needed if c != "full"]
    if not non_full_configs:
        return

    subset_dir = "output/subsets_cifar100"
    missing = []
    for c in non_full_configs:
        path = os.path.join(subset_dir, f"{c}_{mode}.npy")
        alt = os.path.join(subset_dir, f"{c}.npy")
        if not (os.path.exists(path) or os.path.exists(alt)):
            missing.append(c)

    if missing:
        print(f"\n[INFO] Faltan subconjuntos para CIFAR-100: {missing}")
        print("Generando subconjuntos con generate_subsets_fast.py...")
        cmd = [
            sys.executable, "generate_subsets_fast.py",
            "--dataset", "cifar100",
            "--mode", mode,
            "--selection_mode", "unsupervised",
        ]
        run_command(cmd, "Generando subconjuntos SAS y baselines para CIFAR-100")


def print_summary_table():
    cmd = [sys.executable, "show_cifar100_results.py"]
    if os.path.exists("show_cifar100_results.py"):
        subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="Runner integral del Benchmark CIFAR-100")
    parser.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=ALL_METHODS,
                        help="Metodos SSL a ejecutar")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Configuraciones de datos a ejecutar")
    parser.add_argument("--full_matrix", action="store_true",
                        help="Ejecutar la matriz completa (Full + 90%% + 80%% + 60%% + 40%%)")
    parser.add_argument("--mode", choices=["debug", "full"], default="full",
                        help="Modo de ejecucion")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Numero de workers del DataLoader (por defecto 2 para evitar errores IPC en Windows)")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Forzar reentrenamiento aunque el checkpoint o evaluacion ya existan")
    args = parser.parse_args()

    if args.full_matrix:
        data_configs = ALL_DATA_CONFIGS
    elif args.configs is not None:
        data_configs = args.configs
    else:
        # Por defecto: evaluacion full y retencion al 80%
        data_configs = QUICK_CONFIGS

    print(f"\n{'#'*75}")
    print(f"BENCHMARK CIFAR-100 DUAL-DATASET EVALUATION")
    print(f"Metodos: {args.methods}")
    print(f"Configuraciones: {data_configs}")
    print(f"Total combinaciones: {len(args.methods) * len(data_configs)}")
    print(f"Directorio Checkpoints: output/checkpoints_cifar100/")
    print(f"Directorio Resultados:  output/results_cifar100/")
    print(f"{'#'*75}\n")

    os.makedirs("output/checkpoints_cifar100", exist_ok=True)
    os.makedirs("output/results_cifar100", exist_ok=True)
    os.makedirs("output/subsets_cifar100", exist_ok=True)

    # 1. Asegurar subconjuntos
    ensure_subsets_exist(data_configs, mode=args.mode)

    total_start = time.perf_counter()
    completed_runs = 0

    # 2. Iterar por metodo y configuracion
    for method in args.methods:
        for d_cfg in data_configs:
            print(f"\n{'>'*75}")
            print(f">>> MODELO: {method.upper()} | CONFIGURACION: {d_cfg}")
            print(f"{'>'*75}")

            eval_json = f"output/results_cifar100/eval_{method}_{d_cfg}.json"
            ckpt_path = f"output/checkpoints_cifar100/{method}_{d_cfg}_final.pt"

            if os.path.exists(eval_json) and not args.force_retrain:
                print(f"[SKIP] Evaluacion ya existe en: {eval_json}")
                completed_runs += 1
                continue

            # Paso A: Preentrenamiento SSL (si no existe el checkpoint)
            if not os.path.exists(ckpt_path) or args.force_retrain:
                cmd_train = [
                    sys.executable, "train_ssl_final.py",
                    "--method", method,
                    "--data_config", d_cfg,
                    "--dataset", "cifar100",
                    "--mode", args.mode,
                    "--num_workers", str(args.num_workers),
                ]
                run_command(cmd_train, f"SSL Pretraining: {method} ({d_cfg}) en CIFAR-100")
            else:
                print(f"[INFO] Checkpoint existente: {ckpt_path}")

            # Paso B: Evaluacion Lineal
            cmd_eval = [
                sys.executable, "eval_linear.py",
                "--method", method,
                "--data_config", d_cfg,
                "--dataset", "cifar100",
                "--checkpoint", ckpt_path,
                "--mode", args.mode,
            ]
            run_command(cmd_eval, f"Linear Evaluation: {method} ({d_cfg}) en CIFAR-100")
            completed_runs += 1

    total_time = time.perf_counter() - total_start
    print(f"\n{'#'*75}")
    print(f"BENCHMARK CIFAR-100 COMPLETADO")
    print(f"Corridas procesadas: {completed_runs}")
    print(f"Tiempo total transcurrido: {total_time/60:.2f} minutos ({total_time/3600:.2f} horas)")
    print(f"{'#'*75}\n")

    print_summary_table()


if __name__ == "__main__":
    main()
