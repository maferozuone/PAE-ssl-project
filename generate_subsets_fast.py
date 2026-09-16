"""generate_subsets_fast.py
Genera subconjuntos de datos usando SAS (Subsets that maximize Augmentation Similarity)
y baselines aleatorios con porcentajes de conservacion controlados (90%, 80%, 60%, 40%).

MODOS DE SELECCION (--selection_mode):
  - unsupervised (DEFAULT / Puro SSL): Estima clases latentes mediante K-Means sobre
    los embeddings del proxy, SIN usar etiquetas reales. Cumple estrictamente con el
    paradigma auto-supervisado (Joshi & Mirzasoleiman, ICML 2023, Sec. 4.5).
  - oracle: Usa las etiquetas de clase reales de Food-101 para la seleccion (baseline
    de cota superior / supervised proxy).
  - both: Genera tanto los subconjuntos no supervisados como los oraculos.

Uso:
  python generate_subsets_fast.py --mode debug --use_dummy --selection_mode unsupervised
  python generate_subsets_fast.py --mode full --selection_mode unsupervised
  python generate_subsets_fast.py --mode full --selection_mode both
"""

import argparse
import os
import time
import numpy as np

from config_final import get_config
from data_utils import load_food101, generar_dataset_dummy, get_eval_transform
from proxy_model import ProxyModel, compute_embeddings
from sas_selection_fast import (
    approximate_latent_classes,
    sas_select_from_labels_fast,
    random_uniform_subset,
    random_stratified_subset_exact,
)


def main():
    parser = argparse.ArgumentParser(description="Generador acelerado de subconjuntos SAS y baselines")
    parser.add_argument("--mode", choices=["debug", "full"], default="debug")
    parser.add_argument("--use_dummy", action="store_true", help="Usar dataset sintetico para pruebas")
    parser.add_argument("--selection_mode", choices=["unsupervised", "oracle", "both"], default="unsupervised",
                        help="Modo de seleccion: 'unsupervised' (K-Means, sin etiquetas), 'oracle' o 'both'")
    parser.add_argument("--n_clusters", type=int, default=101, help="Numero de clusters latentes para K-Means")
    parser.add_argument("--force_recompute_embeddings", action="store_true")
    parser.add_argument("--proxy_checkpoint", type=str, default=None,
                        help="Ruta opcional a checkpoint SSL para usar como proxy")
    parser.add_argument("--refine", action="store_true", help="Activar refinamiento por swaps (mas lento)")
    args = parser.parse_args()

    cfg = get_config(args.mode)
    print(f"\n{'='*75}")
    print(f"GENERACION DE SUBCONJUNTOS SAS")
    print(f"Modo: {cfg.mode} | Seleccion: {args.selection_mode} | Dispositivo: {cfg.device}")
    print(f"{'='*75}")

    dataset = generar_dataset_dummy(cfg) if args.use_dummy else load_food101(cfg, split="train")
    n_samples = len(dataset)
    print(f"Dataset cargado: {n_samples} muestras")

    emb_path = os.path.join(cfg.subset_dir, f"embeddings_{cfg.mode}.npy")
    labels_path = os.path.join(cfg.subset_dir, f"labels_{cfg.mode}.npy")

    # 1. Cargar o calcular embeddings con el modelo proxy
    if os.path.exists(emb_path) and os.path.exists(labels_path) and not args.force_recompute_embeddings:
        print(f"Cargando embeddings precalculados desde {emb_path} ...")
        embeddings = np.load(emb_path).astype(np.float32)
        labels = np.load(labels_path)
    else:
        print("Calculando embeddings con el modelo proxy...")
        proxy = ProxyModel(cfg.proxy_backbone, pretrained=(args.proxy_checkpoint is None))
        if args.proxy_checkpoint:
            print(f"Cargando checkpoint SSL como proxy: {args.proxy_checkpoint}")
            proxy.load_partial_checkpoint(args.proxy_checkpoint, device="cpu")

        embeddings, labels = compute_embeddings(
            proxy, dataset, get_eval_transform(cfg.image_size), cfg.device,
            batch_size=cfg.batch_size, num_workers=cfg.num_workers
        )
        embeddings = embeddings.astype(np.float32)
        np.save(emb_path, embeddings)
        np.save(labels_path, labels)
        print(f"Embeddings guardados: {embeddings.shape} en {emb_path}")

    # 2. Si se requiere modo no supervisado, calcular clases latentes K-Means una sola vez
    latent_labels = None
    if args.selection_mode in ("unsupervised", "both"):
        n_clusters = min(args.n_clusters, n_samples)
        latent_labels_path = os.path.join(cfg.subset_dir, f"latent_clusters_k{n_clusters}_{cfg.mode}.npy")
        if os.path.exists(latent_labels_path) and not args.force_recompute_embeddings:
            print(f"Cargando clusters latentes K-Means existentes desde {latent_labels_path} ...")
            latent_labels = np.load(latent_labels_path)
        else:
            print(f"Agrupando {n_samples} embeddings en {n_clusters} clusters latentes (K-Means, SIN etiquetas)...")
            latent_labels = approximate_latent_classes(embeddings, n_clusters=n_clusters, seed=42)
            np.save(latent_labels_path, latent_labels)
            print(f"Clusters latentes guardados en {latent_labels_path}")

    # 3. Iterar por cada nivel de retencion/reduccion
    for reduction, keep_fraction in zip(cfg.sas_reduction_percentages, cfg.sas_subset_fractions):
        target_budget = round(n_samples * keep_fraction)
        keep_pct = int(keep_fraction * 100)
        print(f"\n{'='*75}")
        print(f"Objetivo: Reduccion {reduction}% | Conservando {keep_pct}% ({target_budget}/{n_samples} imagenes)")
        print(f"{'='*75}")

        # A. Seleccion No Supervisada (Pure SSL)
        if args.selection_mode in ("unsupervised", "both"):
            print(f"--> [SAS Unsupervised] Seleccionando {target_budget} muestras sobre clusters latentes...")
            sas_unsupervised = sas_select_from_labels_fast(
                embeddings, latent_labels,
                total_budget=target_budget,
                device=cfg.device,
                refine=args.refine,
                refine_iters=2,
                verbose=True
            )
            # Baseline aleatorio uniforme sin etiquetas
            random_unif = random_uniform_subset(n_samples, keep_fraction, seed=42)

            # Guardar subconjuntos principales
            sas_path = os.path.join(cfg.subset_dir, f"sas_keep_{keep_pct}pct_{cfg.mode}.npy")
            rand_path = os.path.join(cfg.subset_dir, f"random_keep_{keep_pct}pct_{cfg.mode}.npy")
            np.save(sas_path, sas_unsupervised["selected_indices"])
            np.save(rand_path, random_unif)

            print(f"  [OK] SAS No Supervisado guardado ({len(sas_unsupervised['selected_indices'])}): {sas_path}")
            print(f"  [OK] Random Uniforme guardado ({len(random_unif)}): {rand_path}")

        # B. Seleccion Supervisada Oracle (Baseline comparativo)
        if args.selection_mode in ("oracle", "both"):
            print(f"--> [SAS Oracle] Seleccionando {target_budget} muestras usando etiquetas reales...")
            sas_oracle = sas_select_from_labels_fast(
                embeddings, labels,
                total_budget=target_budget,
                device=cfg.device,
                refine=args.refine,
                refine_iters=2,
                verbose=True
            )
            random_strat = random_stratified_subset_exact(labels, keep_fraction, seed=42)

            oracle_sas_path = os.path.join(cfg.subset_dir, f"sas_oracle_keep_{keep_pct}pct_{cfg.mode}.npy")
            oracle_rand_path = os.path.join(cfg.subset_dir, f"random_strat_keep_{keep_pct}pct_{cfg.mode}.npy")
            np.save(oracle_sas_path, sas_oracle["selected_indices"])
            np.save(oracle_rand_path, random_strat)

            print(f"  [OK] SAS Oracle guardado ({len(sas_oracle['selected_indices'])}): {oracle_sas_path}")
            print(f"  [OK] Random Estratificado guardado ({len(random_strat)}): {oracle_rand_path}")

            if args.selection_mode == "oracle":
                # Si solo se pidio oracle, crear tambien alias sas_keep_...
                np.save(os.path.join(cfg.subset_dir, f"sas_keep_{keep_pct}pct_{cfg.mode}.npy"), sas_oracle["selected_indices"])
                np.save(os.path.join(cfg.subset_dir, f"random_keep_{keep_pct}pct_{cfg.mode}.npy"), random_strat)

    print(f"\n{'='*75}")
    print(f"Generacion completada con exito en '{cfg.subset_dir}'.")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
