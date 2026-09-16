"""
sas_selection_fast.py

Version optimizada de SAS para Food-101.

Cambios principales respecto a sas_selection.py:
1. Usa similitud coseno mediante multiplicacion matricial de PyTorch.
2. Ejecuta el calculo de ganancias marginales en GPU cuando CUDA esta disponible.
3. Actualiza el vector de mejor similitud incrementalmente, sin recalcular
   la funcion completa para cada candidato.
4. Mantiene refinamiento opcional por swaps, desactivado por defecto porque
   puede ser costoso y no debe bloquear la generacion de subconjuntos.
5. Puede usar etiquetas reales o k-means para definir clases latentes.
6. Usa nombres explicitamente asociados a datos conservados: keep_90pct,
   keep_80pct, keep_60pct y keep_40pct.

La GPU acelera operaciones matriciales y vectorizadas. La decision greedy sigue
siendo secuencial, pero cada iteracion evalua los candidatos en paralelo.
"""

import time
import numpy as np
import torch
from sklearn.cluster import KMeans


@torch.no_grad()
def greedy_facility_location_gpu(embeddings, budget, device=None, verbose=False):
    """
    Seleccion greedy eficiente dentro de una clase.

    embeddings: array (n, d), preferiblemente normalizado.
    budget: cantidad exacta de elementos a seleccionar.
    device: 'cuda', 'cpu' o None para autodeteccion.

    La funcion objetivo es:
        F(S) = sum_i max_{j in S} sim(i, j)

    Como los embeddings estan normalizados, sim(i,j) es producto punto,
    equivalente a similitud coseno.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n = embeddings.shape[0]
    budget = min(int(budget), n)
    if budget <= 0:
        return []

    x = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    x = torch.nn.functional.normalize(x, dim=1)

    # Matriz de similitud completa dentro de la clase.
    # Para 750 x 750 ocupa aproximadamente 2.25 MB en float32.
    similarity = x @ x.T

    selected = []
    remaining = torch.ones(n, dtype=torch.bool, device=device)
    best_similarity = torch.full((n,), -torch.inf, device=device)

    for step in range(budget):
        candidate_best = torch.maximum(best_similarity[:, None], similarity)
        gains = (candidate_best - best_similarity[:, None]).sum(dim=0)
        gains = torch.where(remaining, gains, torch.full_like(gains, -torch.inf))

        selected_idx = int(torch.argmax(gains).item())
        selected.append(selected_idx)
        remaining[selected_idx] = False
        best_similarity = torch.maximum(best_similarity, similarity[:, selected_idx])

        if verbose and ((step + 1) % 25 == 0 or step + 1 == budget):
            print(f"      greedy: {step + 1}/{budget}")

    return selected


@torch.no_grad()
def facility_location_value_gpu(embeddings, selected_indices, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if len(selected_indices) == 0:
        return 0.0

    x = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    x = torch.nn.functional.normalize(x, dim=1)
    selected = x[selected_indices]
    similarity = x @ selected.T
    return float(similarity.max(dim=1).values.sum().item())


@torch.no_grad()
def local_search_refine_gpu(embeddings, initial_indices, max_iters=2, device=None,
                            verbose=False):
    """
    Refinamiento por swaps con presupuesto fijo.

    Esta version limita el numero de rondas para evitar que el refinamiento
    consuma mas tiempo que la seleccion principal. Se puede activar despues
    para un estudio de ablation.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n = embeddings.shape[0]
    selected = list(initial_indices)
    budget = len(selected)
    if budget <= 1 or max_iters <= 0:
        return sorted(selected)

    x = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    x = torch.nn.functional.normalize(x, dim=1)
    similarity = x @ x.T

    selected_set = set(selected)

    def objective(indices):
        return float(similarity[:, indices].max(dim=1).values.sum().item())

    current_value = objective(selected)

    for iteration in range(max_iters):
        improved = False
        outside = [i for i in range(n) if i not in selected_set]

        for pos, old_idx in enumerate(list(selected)):
            candidate_indices = list(selected)
            candidate_indices[pos] = outside[0] if outside else old_idx
            best_candidate = None
            best_value = current_value

            # Evaluamos los candidatos de intercambio en bloques vectorizados.
            if outside:
                base_without = [i for j, i in enumerate(selected) if j != pos]
                base_sim = similarity[:, base_without].max(dim=1).values if base_without else torch.full((n,), -torch.inf, device=device)
                outside_tensor = torch.as_tensor(outside, dtype=torch.long, device=device)
                candidate_values = torch.maximum(base_sim[:, None], similarity[:, outside_tensor]).sum(dim=0)
                max_pos = int(torch.argmax(candidate_values).item())
                max_value = float(candidate_values[max_pos].item())

                if max_value > best_value + 1e-7:
                    best_candidate = outside[max_pos]
                    best_value = max_value

            if best_candidate is not None:
                selected_set.remove(old_idx)
                selected_set.add(best_candidate)
                selected[pos] = best_candidate
                current_value = best_value
                improved = True
                break

        if verbose:
            print(f"      refine: ronda {iteration + 1}/{max_iters}, mejora={improved}")
        if not improved:
            break

    return sorted(selected)


def approximate_latent_classes(embeddings, n_clusters, seed=42, use_minibatch=None):
    """
    Aproxima las clases latentes agrupando embeddings con K-Means (SIN etiquetas).
    Para datasets grandes (>10k muestras) usa MiniBatchKMeans para acelerar y ahorrar memoria.
    """
    n = len(embeddings)
    n_clusters = min(int(n_clusters), n)
    if use_minibatch is None:
        use_minibatch = (n >= 10000)

    if use_minibatch:
        from sklearn.cluster import MiniBatchKMeans
        model = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=seed, batch_size=2048, n_init=3
        )
    else:
        model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)

    return model.fit_predict(embeddings)


def random_uniform_subset(n_total, keep_fraction, seed=42):
    """
    Muestreo aleatorio UNIFORME sin reemplazo sobre todo el dataset (sin estratificar por etiquetas).
    Es el baseline aleatorio estricto para aprendizaje auto-supervisado puro.
    """
    rng = np.random.default_rng(seed)
    total_budget = int(round(n_total * keep_fraction))
    total_budget = min(total_budget, n_total)
    chosen = rng.choice(n_total, size=total_budget, replace=False)
    return np.array(sorted(chosen), dtype=np.int64)


def sas_select_unsupervised_fast(embeddings, total_budget, n_clusters=101,
                                 seed=42, device=None, refine=False, refine_iters=2,
                                 verbose=True):
    """
    Pipeline SAS 100% No Supervisado (sin fuga de etiquetas).
    1. Estima clases latentes agrupando embeddings mediante K-Means.
    2. Distribuye el presupuesto proporcionalmente entre clusters latentes.
    3. Aplica Greedy Facility Location en GPU dentro de cada cluster latente.
    """
    start = time.perf_counter()
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = len(embeddings)
    total_budget = min(int(total_budget), n)

    if verbose:
        print(f"\n  [SAS Unsupervised] Calculando {n_clusters} clusters latentes con K-Means...")
    latent_labels = approximate_latent_classes(embeddings, n_clusters, seed=seed)

    result = sas_select_from_labels_fast(
        embeddings, latent_labels, total_budget, device=device,
        refine=refine, refine_iters=refine_iters, verbose=verbose,
        elapsed_start=start
    )
    result["is_unsupervised"] = True
    result["n_clusters"] = n_clusters
    return result


def sas_full_pipeline_fast(embeddings, subset_fraction, n_clusters=101,
                           seed=42, device=None, use_kmeans=True,
                           refine=False, refine_iters=2, verbose=True):
    """
    Pipeline SAS rapido con tamano objetivo exacto.
    Por defecto use_kmeans=True (modo auto-supervisado estricto).
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = len(embeddings)
    total_budget = int(round(n * subset_fraction))

    if use_kmeans:
        return sas_select_unsupervised_fast(
            embeddings, total_budget, n_clusters=n_clusters,
            seed=seed, device=device, refine=refine,
            refine_iters=refine_iters, verbose=verbose
        )
    else:
        raise ValueError(
            "Para seleccion con etiquetas supervisadas, usa sas_select_from_labels_fast(embeddings, labels, ...)."
        )


def allocate_budget_by_class(labels, total_budget):
    """
    Reparto proporcional entero del presupuesto entre clases.
    Garantiza que la suma sea exactamente total_budget siempre que total_budget >= numero de clases.
    """
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    n = len(labels)
    total_budget = min(int(total_budget), n)

    raw = counts / n * total_budget
    budgets = np.floor(raw).astype(int)

    if total_budget >= len(classes):
        budgets = np.maximum(budgets, 1)

    budgets = np.minimum(budgets, counts)
    current = int(budgets.sum())

    remainders = raw - np.floor(raw)
    order = np.argsort(-remainders)

    while current < total_budget:
        changed = False
        for pos in order:
            if budgets[pos] < counts[pos]:
                budgets[pos] += 1
                current += 1
                changed = True
                if current == total_budget:
                    break
        if not changed:
            break

    while current > total_budget:
        changed = False
        for pos in order[::-1]:
            min_allowed = 1 if total_budget >= len(classes) else 0
            if budgets[pos] > min_allowed:
                budgets[pos] -= 1
                current -= 1
                changed = True
                if current == total_budget:
                    break
        if not changed:
            break

    return classes, budgets


def sas_select_from_labels_fast(embeddings, labels, total_budget, device=None,
                                refine=False, refine_iters=2, verbose=True,
                                elapsed_start=None):
    start = time.perf_counter() if elapsed_start is None else elapsed_start
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels)
    n = len(embeddings)
    total_budget = min(int(total_budget), n)

    classes, budgets = allocate_budget_by_class(labels, total_budget)
    selected_global = []

    if verbose:
        actual_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  SAS rapido usando dispositivo: {actual_device}")
        print(f"  Presupuesto total: {total_budget}/{n} ({total_budget/n*100:.1f}%)")
        print(f"  Clases: {len(classes)} | Refinamiento: {refine}")

    for class_pos, (class_id, budget) in enumerate(zip(classes, budgets), start=1):
        indices = np.where(labels == class_id)[0]
        class_embeddings = embeddings[indices]

        if verbose:
            print(f"  Clase {class_pos}/{len(classes)} (id={class_id}, "
                  f"n={len(indices)}, budget={budget})")

        local_selected = greedy_facility_location_gpu(
            class_embeddings, budget, device=device, verbose=False
        )

        if refine and len(local_selected) > 1:
            local_selected = local_search_refine_gpu(
                class_embeddings, local_selected, max_iters=refine_iters,
                device=device, verbose=False
            )

        selected_global.extend(indices[local_selected].tolist())

    selected_global = np.array(sorted(selected_global), dtype=np.int64)
    elapsed = time.perf_counter() - start

    if verbose:
        print(f"  Seleccion terminada: {len(selected_global)} elementos "
              f"({len(selected_global)/n*100:.2f}%) en {elapsed/60:.2f} min")

    return {
        "selected_indices": selected_global,
        "latent_labels": labels,
        "fraction_achieved": len(selected_global) / n,
        "n_selected": len(selected_global),
        "n_total": n,
        "elapsed_seconds": elapsed,
        "device": str(device or ("cuda" if torch.cuda.is_available() else "cpu")),
        "refine": refine,
    }


def random_stratified_subset_exact(labels, subset_fraction, seed=42):
    """Baseline aleatorio con exactamente el mismo reparto por clase que SAS."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    total_budget = int(round(len(labels) * subset_fraction))
    classes, budgets = allocate_budget_by_class(labels, total_budget)
    selected = []

    for class_id, budget in zip(classes, budgets):
        indices = np.where(labels == class_id)[0]
        if budget > 0:
            selected.extend(rng.choice(indices, size=budget, replace=False).tolist())

    return np.array(sorted(selected), dtype=np.int64)


if __name__ == "__main__":
    print("sas_selection_fast.py listo.")
    print("Usa embeddings y labels cargados para llamar a sas_select_from_labels_fast().")
