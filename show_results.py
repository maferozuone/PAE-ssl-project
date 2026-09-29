import json
import glob
import os

METHODS = ["align_uniform", "byol", "cpc", "simsiam"]
CONFIGS = [
    "full",
    "sas_keep_90pct",
    "random_keep_90pct",
    "sas_keep_80pct",
    "random_keep_80pct",
    "sas_keep_60pct",
    "random_keep_60pct",
]

def main():
    files = sorted(glob.glob('output/results_final/eval_*.json'))
    print(f"\nEncontrados {len(files)} archivos de evaluacion.\n")

    # 1. Tabla detallada
    header = f"{'Metodo':<16} {'Data Config':<25} {'Best Top-1':>10} {'Best Top-5':>10} {'Epoca':>8} {'Modo':>8}"
    print(header)
    print('-' * 85)

    results_map = {}
    for f in files:
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        method = d.get('method', '?')
        dc = d.get('data_config', '?')
        t1 = d.get('best_top1', 0)
        t5 = d.get('best_top5', 0)
        ep = d.get('best_epoch', 0)
        mode = d.get('mode', '?')
        print(f"{method:<16} {dc:<25} {t1:>9.2f}% {t5:>9.2f}% {ep:>8} {mode:>8}")
        if mode == "full":
            results_map[(method, dc)] = d

    # 2. Matriz consolidada de 28 experimentos
    print("\n" + "=" * 82)
    print("MATRIZ DE EXPERIMENTOS (MODO FULL): 28 CONFIGURACIONES")
    print("=" * 82)
    cols = " | ".join([f"{m:<12}" for m in METHODS])
    print(f"{'Configuracion':<20} | {cols}")
    print("-" * 82)

    done_count = 0
    for c in CONFIGS:
        vals = []
        for m in METHODS:
            if (m, c) in results_map:
                done_count += 1
                t1 = results_map[(m, c)].get("best_top1", 0)
                vals.append(f"{t1:>5.2f}% (OK) ")
            else:
                vals.append(" PENDIENTE  ")
        print(f"{c:<20} | " + " | ".join(vals))

    print("=" * 82)
    pct = (done_count / 28) * 100
    print(f"Completados: {done_count}/28 ({pct:.1f}%) | Faltantes: {28 - done_count}")
    print("=" * 82 + "\n")

if __name__ == "__main__":
    main()
