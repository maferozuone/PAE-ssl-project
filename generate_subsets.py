"""
generate_subsets.py
Script que conecta todo el pipeline previo al entrenamiento SSL:

    Food-101 (o dummy) --> ProxyModel --> embeddings --> SAS --> subconjuntos
                                                       --> aleatorio --> subconjuntos

Genera y GUARDA en disco (carpeta cfg.subset_dir) los indices de:
  - Los subconjuntos SAS para cada fraccion en cfg.sas_subset_fractions (10/20/40/60%)
  - Los subconjuntos aleatorios estratificados del MISMO tamano, para comparacion

USO:
    python generate_subsets.py --mode debug --use_dummy
    python generate_subsets.py --mode full            # en la PC de la universidad

CORRECCION APLICADA (ver notas en data_utils.py y sas_selection.py):
El dataset dummy ahora usa pocas clases simuladas en modo debug, para evitar
el caso limite donde el minimo de 1 elemento/clase domina sobre la fraccion
pedida. Si SAS detecta ese caso limite igualmente (por ejemplo, si usas
Food-101 real con una fraccion de debug extremadamente pequena), se imprime
una advertencia explicita explicando por que el subconjunto resultante es
mas grande de lo pedido.
"""

import argparse
import os
import numpy as np

from config import get_config
from data_utils import (
    load_food101, generar_dataset_dummy, get_eval_transform
)
from proxy_model import ProxyModel, compute_embeddings
from sas_selection import sas_full_pipeline, random_stratified_subset


def get_dataset(cfg, use_dummy):
    """
    Carga Food-101 real, o el dataset dummy si use_dummy=True.
    """
    if use_dummy:
        print("[AVISO] Usando dataset DUMMY (datos sinteticos). "
              "Los resultados NO son validos para el proyecto real, "
              "solo sirven para validar que el pipeline corre sin errores.")
        dataset = generar_dataset_dummy(cfg)
        print(f"   Dataset dummy: {len(dataset)} muestras, {dataset.n_classes} clases dummy")
        return dataset
    else:
        print(f"Cargando Food-101 (split=train, fraction={cfg.debug_fraction})...")
        return load_food101(cfg, split="train")


def compute_or_load_embeddings(cfg, dataset, force_recompute=False):
    """
    Calcula los embeddings del proxy model, o los carga desde disco si ya
    existen (para no recalcular cada vez que se corre el script).
    """
    embeddings_path = os.path.join(cfg.subset_dir, f"embeddings_{cfg.mode}.npy")
    labels_path = os.path.join(cfg.subset_dir, f"labels_{cfg.mode}.npy")

    if os.path.exists(embeddings_path) and os.path.exists(labels_path) and not force_recompute:
        print(f"Cargando embeddings ya calculados desde {embeddings_path} ...")
        embeddings = np.load(embeddings_path)
        labels = np.load(labels_path)
        return embeddings, labels

    print("Calculando embeddings con el modelo proxy (puede tardar)...")
    proxy = ProxyModel(backbone_name=cfg.proxy_backbone, pretrained=True)
    eval_transform = get_eval_transform(cfg.image_size)
    embeddings, labels = compute_embeddings(
        proxy, dataset, eval_transform, cfg.device,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    np.save(embeddings_path, embeddings)
    np.save(labels_path, labels)
    print(f"Embeddings guardados en {embeddings_path}")

    return embeddings, labels


def generate_all_subsets(cfg, embeddings, labels, use_ground_truth_labels=True):
    """
    Genera y guarda los subconjuntos SAS y aleatorios para todas las
    fracciones en cfg.sas_subset_fractions.

    Args:
        use_ground_truth_labels: si True, usa las etiquetas REALES de Food-101
            para aproximar las clases latentes. Si False, usa k-means sobre
            los embeddings (mas fiel al escenario sin ninguna etiqueta
            disponible, ver Sec. 4.5 del paper SAS).
    """
    results_summary = []
    n_unique_labels = len(np.unique(labels))
    print(f"\n[INFO] Numero de clases distintas en el dataset: {n_unique_labels}")
    print(f"[INFO] Numero total de muestras: {len(embeddings)}")
    print(f"[INFO] Promedio de muestras por clase: {len(embeddings)/n_unique_labels:.1f}\n")

    for fraction in cfg.sas_subset_fractions:
        print(f"\n{'='*60}")
        print(f"Generando subconjuntos para fraccion = {fraction*100:.0f}%")
        print(f"{'='*60}")

        # --- SAS ---
        gt_labels = labels if use_ground_truth_labels else None
        sas_result = sas_full_pipeline(
            embeddings, subset_fraction=fraction,
            n_clusters=cfg.num_classes, seed=42,
            ground_truth_labels=gt_labels
        )
        sas_indices = np.array(sas_result["selected_indices"])
        sas_path = os.path.join(cfg.subset_dir, f"sas_{int(fraction*100)}pct_{cfg.mode}.npy")
        np.save(sas_path, sas_indices)

        # --- Aleatorio (mismo tamano, estratificado) ---
        random_indices = np.array(
            random_stratified_subset(labels, subset_fraction=fraction, seed=42)
        )
        random_path = os.path.join(cfg.subset_dir, f"random_{int(fraction*100)}pct_{cfg.mode}.npy")
        np.save(random_path, random_indices)

        print(f"  SAS: {len(sas_indices)} indices guardados en {sas_path}")
        print(f"  Random: {len(random_indices)} indices guardados en {random_path}")

        results_summary.append({
            "fraction_pedida": fraction,
            "sas_n": len(sas_indices),
            "sas_fraction_real": len(sas_indices) / len(embeddings),
            "random_n": len(random_indices),
            "random_fraction_real": len(random_indices) / len(embeddings),
            "budget_dominated": sas_result["budget_dominated_by_min_per_class"],
        })

    return results_summary


def main():
    parser = argparse.ArgumentParser(description="Genera subconjuntos SAS y aleatorios")
    parser.add_argument("--mode", type=str, default="debug", choices=["debug", "full"])
    parser.add_argument("--use_dummy", action="store_true",
                         help="Usar dataset dummy en vez de Food-101 real (solo para pruebas de pipeline)")
    parser.add_argument("--force_recompute_embeddings", action="store_true",
                         help="Recalcular embeddings aunque ya existan en disco")
    parser.add_argument("--use_kmeans_latent_classes", action="store_true",
                         help="Usar k-means en vez de las etiquetas reales para aproximar clases latentes")
    args = parser.parse_args()

    cfg = get_config(args.mode)
    print(f"Configuracion: {cfg}")

    dataset = get_dataset(cfg, use_dummy=args.use_dummy)
    print(f"Dataset cargado: {len(dataset)} muestras")

    embeddings, labels = compute_or_load_embeddings(
        cfg, dataset, force_recompute=args.force_recompute_embeddings
    )
    print(f"Embeddings: {embeddings.shape}, Labels: {labels.shape}")

    summary = generate_all_subsets(
        cfg, embeddings, labels,
        use_ground_truth_labels=not args.use_kmeans_latent_classes
    )

    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print(f"{'='*60}")
    for row in summary:
        flag = " [DOMINADO POR MINIMO/CLASE]" if row["budget_dominated"] else ""
        print(f"  Pedido {row['fraction_pedida']*100:.0f}% -> "
              f"SAS: {row['sas_n']} ({row['sas_fraction_real']*100:.1f}%) | "
              f"Random: {row['random_n']} ({row['random_fraction_real']*100:.1f}%){flag}")

    print(f"\nTodos los subconjuntos se guardaron en: {cfg.subset_dir}")


if __name__ == "__main__":
    main()
