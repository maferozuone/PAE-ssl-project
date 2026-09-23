"""run_experiments.py
Ejecutor por lotes de experimentos SSL para paralelizar entre maquinas (Kaggle, PC de la U, etc.).

Uso:
  # Correr solo metodos asignados al PC de la U:
  python run_experiments.py --methods cpc align_uniform

  # Correr metodos especificos con configs especificas:
  python run_experiments.py --methods cpc --data_configs full sas_keep_60pct

  # Modo rapido de prueba (debug):
  python run_experiments.py --methods simsiam --mode debug
"""

import argparse
import json
import os
import time

from config_final import get_config
import train_ssl_final as train_module

DEFAULT_METHODS = ["simsiam", "byol", "cpc", "align_uniform"]
DEFAULT_CONFIGS = ["full", "sas_keep_80pct", "sas_keep_60pct", "random_keep_60pct"]


def main():
    parser = argparse.ArgumentParser(description="Ejecutor de experimentos SSL por lotes")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS,
        help="Metodos SSL a entrenar (ej: --methods cpc align_uniform)",
    )
    parser.add_argument(
        "--data_configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Configuraciones de datos (ej: --data_configs full sas_keep_60pct)",
    )
    parser.add_argument(
        "--mode",
        choices=["debug", "full"],
        default="full",
        help="Modo de entrenamiento: 'full' (completo) o 'debug' (prueba rapida)",
    )
    parser.add_argument(
        "--use_dummy",
        action="store_true",
        help="Usar dataset dummy para verificar la instalacion",
    )
    args = parser.parse_args()

    cfg = get_config(args.mode)

    print("=" * 70)
    print("EJECUTOR DE EXPERIMENTOS SSL EN PARALELO")
    print(f"Modo:             {args.mode}")
    print(f"Dispositivo:      {cfg.device}")
    print(f"Metodos:          {args.methods}")
    print(f"Data configs:     {args.data_configs}")
    print(f"Total a procesar: {len(args.methods) * len(args.data_configs)}")
    print("=" * 70)

    results = []
    total = len(args.methods) * len(args.data_configs)
    idx = 0

    for method in args.methods:
        for data_config in args.data_configs:
            idx += 1
            ckpt_name = f"{method}_{data_config}_final.pt"
            ckpt_path = os.path.join(cfg.checkpoint_dir, ckpt_name)

            print(f"\n>>> [{idx}/{total}] Metodo: {method.upper()} | Datos: {data_config}")
            print("-" * 60)

            if os.path.exists(ckpt_path):
                print(f"[SKIP] Checkpoint ya existe: {ckpt_path}")
                results.append({
                    "method": method,
                    "data_config": data_config,
                    "status": "skipped",
                    "checkpoint": ckpt_path,
                })
                continue

            t0 = time.perf_counter()
            try:
                train_module.train(method, data_config, mode=args.mode, use_dummy=args.use_dummy)
                elapsed = time.perf_counter() - t0
                print(f"[OK] Completado en {elapsed / 60:.1f} minutos.")
                results.append({
                    "method": method,
                    "data_config": data_config,
                    "status": "done",
                    "elapsed_min": round(elapsed / 60, 2),
                    "checkpoint": ckpt_path,
                })
            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"[ERROR] Fallo: {e}")
                results.append({
                    "method": method,
                    "data_config": data_config,
                    "status": "error",
                    "elapsed_min": round(elapsed / 60, 2),
                    "error": str(e),
                })

    # Guardar resumen de ejecucion
    summary_file = os.path.join(cfg.results_dir, f"batch_summary_{args.mode}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("RESUMEN FINAL DE LA TANDA")
    print("=" * 70)
    for r in results:
        status_tag = {
            "done": "OK",
            "skipped": "OMITIDO (ya existia)",
            "error": "ERROR",
        }.get(r["status"], r["status"])
        mins = f" ({r.get('elapsed_min', 0):.1f} min)" if r["status"] == "done" else ""
        print(f"[{status_tag}] {r['method']:15s} | {r['data_config']:25s}{mins}")

    print(f"\nResumen guardado en: {summary_file}")


if __name__ == "__main__":
    main()
