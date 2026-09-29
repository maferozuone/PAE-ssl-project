import json, glob, os

files = sorted(glob.glob('output/results_final/eval_*.json'))
print(f'Encontrados {len(files)} archivos de evaluacion\n')

header = f"{'Metodo':<16} {'Data Config':<25} {'Best Top-1':>10} {'Best Top-5':>10} {'Epoca':>8} {'Modo':>8}"
print(header)
print('-' * 85)

for f in files:
    with open(f) as fp:
        d = json.load(fp)
    method = d.get('method', '?')
    dc = d.get('data_config', '?')
    t1 = d.get('best_top1', 0)
    t5 = d.get('best_top5', 0)
    ep = d.get('best_epoch', 0)
    mode = d.get('mode', '?')
    print(f"{method:<16} {dc:<25} {t1:>9.2f}% {t5:>9.2f}% {ep:>8} {mode:>8}")
