"""run_experiments.py
Ejecutor integral por lotes de experimentos SSL y Evaluacion Lineal.
Permite entrenar y evaluar automaticamente los modelos faltantes, saltando
aquellos que ya fueron completados.

Uso:
  # Ejecutar todos los experimentos faltantes (preentrenamiento SSL + evaluacion lineal):
  python run_experiments.py

  # Ejecutar metodos especificos:
  python run_experiments.py --methods byol simsiam

  # Ejecutar configuraciones especificas:
  python run_experiments.py --data_configs sas_keep_90pct random_keep_90pct random_keep_80pct

  # Solo preentrenamiento SSL sin evaluacion lineal:
  python run_experiments.py --no_eval

  # Modo rapido de prueba (debug):
  python run_experiments.py --mode debug --use_dummy
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

from config_final import get_config
import train_ssl_final as train_module
import eval_linear as eval_module

DEFAULT_METHODS = ["align_uniform", "byol", "cpc", "simsiam"]
DEFAULT_CONFIGS = [
    "sas_keep_90pct",
    "random_keep_90pct",
    "random_keep_80pct",
    "sas_keep_80pct",
    "sas_keep_60pct",
    "random_keep_60pct",
    "full",
]


def check_and_ensure_subsets(cfg, data_configs, use_dummy=False):
    """Verifica si los subconjuntos requeridos existen en disco."""
    if use_dummy:
        return

    missing_subsets = []
    for dc in data_configs:
        if dc == "full":
            continue
        try:
            train_module.load_indices(cfg, dc)
        except FileNotFoundError:
            missing_subsets.append(dc)

    if missing_subsets:
        print(f"\n[ALERTA] Subconjuntos no encontrados en {cfg.subset_dir}: {missing_subsets}")
        print("Generando subconjuntos faltantes automaticamente con generate_subsets_fast.py...")
        cmd = [
            sys.executable, "generate_subsets_fast.py",
            "--mode", cfg.mode,
            "--selection_mode", "unsupervised"
        ]
        print(f"Ejecutando: {' '.join(cmd)}")
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            raise RuntimeError(f"Error generando subconjuntos con generate_subsets_fast.py (codigo {ret.returncode})")
        print("[OK] Subconjuntos generados exitosamente.\n")


def main():
    parser = argparse.ArgumentParser(description="Ejecutor integral de experimentos SSL + Linear Probing")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS,
        help="Metodos SSL a procesar (ej: --methods byol simsiam)",
    )
    parser.add_argument(
        "--data_configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Configuraciones de datos a procesar",
    )
    parser.add_argument(
        "--mode",
        choices=["debug", "full"],
        default="full",
        help="Modo: 'full' (completo) o 'debug' (prueba rapida)",
    )
    parser.add_argument(
        "--use_dummy",
        action="store_true",
        help="Usar dataset dummy para verificar la instalacion",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Omitir la evaluacion lineal tras el preentrenamiento SSL",
    )
    parser.add_argument(
        "--eval_epochs",
        type=int,
        default=None,
        help="Numero de epocas de evaluacion lineal (default: 100 en full, 3 en debug)",
    )
    parser.add_argument(
        "--force_ssl",
        action="store_true",
        help="Forzar reentrenamiento SSL incluso si el checkpoint existe",
    )
    parser.add_argument(
        "--force_eval",
        action="store_true",
        help="Forzar re-evaluacion lineal incluso si el archivo JSON existe",
    )
    args = parser.parse_args()

    cfg = get_config(args.mode)

    print("=" * 80)
    print("EJECUTOR AUTOMATIZADO DE EXPERIMENTOS SSL + LINEAR PROBING")
    print(f"Modo:             {args.mode}")
    print(f"Dispositivo:      {cfg.device}")
    print(f"Metodos:          {args.methods}")
    print(f"Data configs:     {args.data_configs}")
    print(f"Eval lineal:      {'DESACTIVADA' if args.no_eval else 'ACTIVADA'}")
    print("=" * 80)

    # 1. Verificar existencia de subconjuntos
    check_and_ensure_subsets(cfg, args.data_configs, use_dummy=args.use_dummy)

    results = []
    total = len(args.methods) * len(args.data_configs)
    idx = 0

    for method in args.methods:
        for data_config in args.data_configs:
            idx += 1
            ckpt_name = f"{method}_{data_config}_final.pt"
            ckpt_path = os.path.join(cfg.checkpoint_dir, ckpt_name)
            eval_name = f"eval_{method}_{data_config}.json"
            eval_path = os.path.join(cfg.results_dir, eval_name)

            ssl_exists = os.path.exists(ckpt_path)
            eval_exists = os.path.exists(eval_path)

            print(f"\n{'='*75}")
            print(f">>> [{idx}/{total}] Metodo: {method.upper()} | Datos: {data_config}")
            print(f"    Checkpoint SSL:  {'[EXISTE]' if ssl_exists else '[PENDIENTE]'} -> {ckpt_path}")
            print(f"    Eval Lineal:     {'[EXISTE]' if eval_exists else '[PENDIENTE]'} -> {eval_path}")
            print(f"{'='*75}")

            item_res = {
                "method": method,
                "data_config": data_config,
                "ssl_status": "pending",
                "eval_status": "pending",
                "checkpoint": ckpt_path,
                "eval_json": eval_path,
            }

            # A. PREENTRENAMIENTO SSL
            if ssl_exists and not args.force_ssl:
                print(f"[SKIP SSL] Checkpoint ya existe: {ckpt_name}")
                item_res["ssl_status"] = "skipped"
            else:
                print(f"--> [SSL TRAIN] Entrenando {method.upper()} ({cfg.epochs_ssl} epocas)...")
                t0 = time.perf_counter()
                try:
                    train_module.train(method, data_config, mode=args.mode, use_dummy=args.use_dummy)
                    elapsed = time.perf_counter() - t0
                    print(f"[OK SSL] Completado en {elapsed / 60:.1f} minutos.")
                    item_res["ssl_status"] = "done"
                    item_res["ssl_time_min"] = round(elapsed / 60, 2)
                except Exception as e:
                    elapsed = time.perf_counter() - t0
                    print(f"[ERROR SSL] Fallo preentrenamiento: {e}")
                    item_res["ssl_status"] = "error"
                    item_res["ssl_error"] = str(e)
                    results.append(item_res)
                    continue

            # B. EVALUACION LINEAL
            if not args.no_eval:
                if eval_exists and not args.force_eval:
                    print(f"[SKIP EVAL] Evaluacion ya existe: {eval_name}")
                    item_res["eval_status"] = "skipped"
                else:
                    if not os.path.exists(ckpt_path):
                        print(f"[ERROR EVAL] No se encontro checkpoint para evaluar: {ckpt_path}")
                        item_res["eval_status"] = "missing_checkpoint"
                    else:
                        print(f"--> [LINEAR EVAL] Iniciando Linear Probing sobre Food-101...")
                        t0 = time.perf_counter()
                        try:
                            eval_summary = eval_module.train_linear_eval(
                                cfg,
                                ckpt_path,
                                epochs=args.eval_epochs,
                                lr=0.1,
                                optimizer_name="sgd",
                                use_dummy=args.use_dummy,
                            )
                            elapsed = time.perf_counter() - t0
                            top1 = eval_summary.get("best_top1", 0.0)
                            top5 = eval_summary.get("best_top5", 0.0)
                            print(f"[OK EVAL] Mejor Top-1: {top1:.2f}% | Top-5: {top5:.2f}% ({elapsed / 60:.1f} min)")
                            item_res["eval_status"] = "done"
                            item_res["eval_time_min"] = round(elapsed / 60, 2)
                            item_res["best_top1"] = top1
                            item_res["best_top5"] = top5
                        except Exception as e:
                            elapsed = time.perf_counter() - t0
                            print(f"[ERROR EVAL] Fallo evaluacion lineal: {e}")
                            item_res["eval_status"] = "error"
                            item_res["eval_error"] = str(e)

            results.append(item_res)

    # Guardar resumen de ejecucion
    summary_file = os.path.join(cfg.results_dir, f"batch_summary_{args.mode}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("RESUMEN DE LA EJECUCION")
    print("=" * 80)
    for r in results:
        ssl_s = r["ssl_status"].upper()
        eval_s = r["eval_status"].upper()
        m_s = f"{r['method']:15s} | {r['data_config']:22s} | SSL: {ssl_s:7s} | EVAL: {eval_s:7s}"
        if "best_top1" in r:
            m_s += f" | Top-1: {r['best_top1']:.2f}%"
        print(m_s)

    print(f"\nResumen guardado en: {summary_file}")
    print("Puedes ver la matriz actualizada ejecutando: python show_results.py\n")


if __name__ == "__main__":
    main()
