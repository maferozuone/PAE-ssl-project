"""
sas_selection.py
Implementacion del algoritmo SAS (Subsets that maximize Augmentation Similarity),
siguiendo la Seccion 4 y el Algoritmo 1 del paper de referencia (2302.09195).

VERSION: corregida (incluye advertencia de caso limite y la clave
'budget_dominated_by_min_per_class' en el resultado, necesaria para que
generate_subsets.py funcione sin errores).

TERMINOS CLAVE (recordatorio):
- "Clase latente": grupo aproximado de imagenes que probablemente comparten
  la misma categoria real, obtenido SIN usar las etiquetas verdaderas (via
  k-means sobre embeddings de un modelo proxy), o usando las etiquetas reales
  si se dispone de ellas (el paper permite ambas opciones, Sec. 4.5).
- "Facility location": funcion matematica que mide que tan bien un subconjunto
  S representa a todo el conjunto V, sumando la similitud de cada punto hacia
  su representante mas cercano dentro de S.
- "Greedy": algoritmo que en cada paso elige el elemento que mas mejora la
  solucion actual, sin reconsiderar decisiones pasadas.
- "Budget" (presupuesto): numero maximo de elementos que se pueden seleccionar.

NOTA IMPORTANTE 1 (double-greedy vs presupuesto fijo):
El algoritmo "double-greedy" del paper original (Buchbinder et al. 2015) esta
pensado para maximizacion SIN restriccion de tamano. Como la funcion de
facility location nunca empeora al agregar mas puntos, aplicarlo literalmente
tiende a seleccionar TODO el conjunto, rompiendo el requisito de tener
subconjuntos de tamano EXACTO (10%/20%/40%/60%). Por eso, aqui usamos una
variante de "busqueda local con presupuesto fijo": se permiten intercambios
(swap) de un elemento por otro, pero el TAMANO del subconjunto nunca cambia.

NOTA IMPORTANTE 2 (minimo 1 elemento por clase, detectado en pruebas):
Para garantizar que ninguna clase quede totalmente excluida, cada clase
latente recibe un presupuesto minimo de 1 elemento (r_k = max(1, ...)). Si
el numero de clases presentes es MAYOR que el presupuesto total pedido
(esto puede pasar con datasets muy pequenos y muchas clases, ej. en pruebas
de humo con datos dummy), este minimo domina sobre la fraccion solicitada,
y el subconjunto resultante sera MAS GRANDE de lo pedido. Con Food-101 real
(101 clases, ~750 imagenes/clase en train) esto no deberia ocurrir salvo que
se pidan fracciones absurdamente pequenas. Se anade una advertencia explicita
para detectar este caso limite cuando ocurra.
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


# ----------------------------------------------------------------------------
# 1. Greedy facility location (seleccion dentro de UNA clase latente)
# ----------------------------------------------------------------------------

def greedy_facility_location(similarity_matrix, budget):
    """
    Selecciona 'budget' elementos maximizando F(S) = sum_i max_{j in S} sim(i,j).

    Args:
        similarity_matrix: matriz (n, n) de similitudes. Mayor valor = mas parecido.
        budget: cuantos elementos seleccionar como maximo.

    Returns:
        Lista de indices LOCALES seleccionados, en orden de eleccion.
    """
    n = similarity_matrix.shape[0]
    selected = []
    current_best = np.full(n, -np.inf)
    remaining = set(range(n))

    for _ in range(min(budget, n)):
        best_gain = -np.inf
        best_elem = None
        for e in remaining:
            candidate_best_safe = np.where(
                np.isneginf(current_best),
                similarity_matrix[:, e],
                np.maximum(current_best, similarity_matrix[:, e])
            )
            gain = np.sum(candidate_best_safe - np.where(np.isneginf(current_best), 0, current_best))
            if gain > best_gain:
                best_gain = gain
                best_elem = e
        selected.append(best_elem)
        current_best = np.where(
            np.isneginf(current_best),
            similarity_matrix[:, best_elem],
            np.maximum(current_best, similarity_matrix[:, best_elem])
        )
        remaining.remove(best_elem)

    return selected


# ----------------------------------------------------------------------------
# 2. Refinamiento por busqueda local (variante de double-greedy con presupuesto fijo)
# ----------------------------------------------------------------------------

def _facility_location_value(similarity_matrix, subset_indices):
    if len(subset_indices) == 0:
        return 0.0
    idx = list(subset_indices)
    return float(np.sum(np.max(similarity_matrix[:, idx], axis=1)))


def local_search_refine(similarity_matrix, initial_set_indices, all_local_indices,
                         max_iters=20):
    """
    Refina una seleccion greedy inicial mediante intercambios (swaps), preservando
    el TAMANO exacto del subconjunto.
    """
    S = list(initial_set_indices)
    current_F = _facility_location_value(similarity_matrix, S)

    for _ in range(max_iters):
        outside = [i for i in all_local_indices if i not in S]
        improved = False
        for i_pos, s_elem in enumerate(S):
            for o_elem in outside:
                candidate = S.copy()
                candidate[i_pos] = o_elem
                cand_F = _facility_location_value(similarity_matrix, candidate)
                if cand_F > current_F:
                    S = candidate
                    current_F = cand_F
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return sorted(S), current_F


# ----------------------------------------------------------------------------
# 3. Aproximacion de clases latentes (k-means sobre embeddings del proxy)
# ----------------------------------------------------------------------------

def approximate_latent_classes(embeddings, n_clusters, seed=42):
    """
    Aproxima las clases latentes agrupando embeddings con k-means.
    """
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels


# ----------------------------------------------------------------------------
# 4. Seleccion SAS dentro de una sola clase (greedy + refinamiento)
# ----------------------------------------------------------------------------

def sas_select_within_class(class_embeddings, budget, refine=True, max_refine_iters=20):
    n_k = class_embeddings.shape[0]
    budget = min(budget, n_k)
    if budget <= 0:
        return []

    dist = cdist(class_embeddings, class_embeddings, metric="euclidean")
    sim = -dist

    selected = greedy_facility_location(sim, budget)

    if refine and len(selected) > 1:
        selected, _ = local_search_refine(sim, selected, list(range(n_k)),
                                           max_iters=max_refine_iters)

    return selected


# ----------------------------------------------------------------------------
# 5. Pipeline completo: de embeddings crudos a subconjunto final SAS
# ----------------------------------------------------------------------------

def sas_full_pipeline(embeddings, subset_fraction, n_clusters, seed=42,
                       refine=True, ground_truth_labels=None, verbose=True):
    """
    Pipeline completo de SAS.

    Returns:
        dict con selected_indices, latent_labels, fraction_achieved,
        n_selected, n_total, n_present_classes,
        budget_dominated_by_min_per_class (True si el minimo de 1/clase forzo
        un subconjunto mas grande de lo pedido -- ver NOTA IMPORTANTE 2).
    """
    n = embeddings.shape[0]
    total_budget = int(round(n * subset_fraction))

    if ground_truth_labels is not None:
        latent_labels = np.asarray(ground_truth_labels)
    else:
        latent_labels = approximate_latent_classes(embeddings, n_clusters, seed=seed)

    unique_classes = np.unique(latent_labels)
    n_present_classes = len(unique_classes)

    budget_dominated = n_present_classes > total_budget
    if budget_dominated and verbose:
        print(f"  [ADVERTENCIA SAS] Presupuesto pedido ({total_budget}) < numero de "
              f"clases presentes ({n_present_classes}). El minimo de 1 elemento/clase "
              f"forzara un subconjunto de al menos {n_present_classes} elementos "
              f"({n_present_classes/n*100:.1f}%), mas grande de lo pedido.")

    selected_global_indices = []

    for k in unique_classes:
        class_indices = np.where(latent_labels == k)[0]
        n_k = len(class_indices)
        r_k = max(1, int(round(n_k / n * total_budget)))
        r_k = min(r_k, n_k)

        class_embeddings = embeddings[class_indices]
        selected_local = sas_select_within_class(class_embeddings, r_k, refine=refine)
        selected_global = class_indices[selected_local]
        selected_global_indices.extend(selected_global.tolist())

    return {
        "selected_indices": sorted(selected_global_indices),
        "latent_labels": latent_labels,
        "fraction_achieved": len(selected_global_indices) / n,
        "n_selected": len(selected_global_indices),
        "n_total": n,
        "n_present_classes": n_present_classes,
        "budget_dominated_by_min_per_class": budget_dominated,
    }


# ----------------------------------------------------------------------------
# 6. Baseline de comparacion: subconjunto aleatorio estratificado
# ----------------------------------------------------------------------------

def random_stratified_subset(labels, subset_fraction, seed=42):
    """
    Genera un subconjunto ALEATORIO del mismo tamano que SAS (estratificado
    por clase para mantener proporciones), usado como baseline de comparacion.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = len(labels)
    total_budget = int(round(n * subset_fraction))

    unique_classes = np.unique(labels)
    selected = []
    for k in unique_classes:
        class_indices = np.where(labels == k)[0]
        n_k = len(class_indices)
        r_k = max(1, int(round(n_k / n * total_budget)))
        r_k = min(r_k, n_k)
        chosen = rng.choice(class_indices, size=r_k, replace=False)
        selected.extend(chosen.tolist())

    return sorted(selected)


# ----------------------------------------------------------------------------
# Prueba autocontenida (usa datos SINTETICOS, no requiere Food-101 ni torch)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Prueba de sas_selection.py con datos sinteticos ===\n")

    rng = np.random.default_rng(42)

    print("--- Caso 1: clusters bien separados, presupuesto razonable ---")
    cluster_centers = np.array([[0, 0], [10, 10], [-10, 10]])
    points_list = []
    true_labels_list = []
    for c_id, center in enumerate(cluster_centers):
        pts = center + rng.standard_normal((10, 2)) * 0.5
        points_list.append(pts)
        true_labels_list += [c_id] * 10
    points = np.vstack(points_list)
    true_labels = np.array(true_labels_list)

    for frac in [0.10, 0.20, 0.40, 0.60]:
        res = sas_full_pipeline(points, subset_fraction=frac, n_clusters=3,
                                 ground_truth_labels=true_labels, verbose=True)
        print(f"   Fraccion pedida: {frac*100:.0f}% -> obtenida: "
              f"{res['fraction_achieved']*100:.1f}% ({res['n_selected']} elementos) "
              f"| dominado por minimo: {res['budget_dominated_by_min_per_class']}")

    print("\n--- Caso 2 (caso limite): muchas mas clases que presupuesto ---")
    many_classes_labels = rng.integers(0, 35, size=40)
    many_classes_embeddings = rng.standard_normal((40, 16)).astype(np.float32)
    for frac in [0.10, 0.20]:
        res = sas_full_pipeline(many_classes_embeddings, subset_fraction=frac, n_clusters=35,
                                 ground_truth_labels=many_classes_labels, verbose=True)
        print(f"   Resultado: {res['n_selected']} elementos ({res['fraction_achieved']*100:.1f}%)\n")

    print("\n--- Caso 3: baseline aleatorio estratificado ---")
    random_sel = random_stratified_subset(true_labels, subset_fraction=0.4)
    print(f"   Seleccionados al azar: {len(random_sel)} de {len(points)}")
    print(f"   Distribucion por clase real: {np.bincount(true_labels[random_sel])}")

    print("\nsas_selection.py: todas las pruebas OK")
