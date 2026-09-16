"""
ssl_losses.py
Las 4 funciones de perdida correspondientes a cada metodo SSL, ya validadas
matematicamente con numpy antes de traducirlas a PyTorch (ver conversacion
del proyecto: se confirmo que cada formula se comporta como predicen los
papers originales, usando casos de prueba controlados).

TERMINOS CLAVE (recordatorio):
- "Stop-gradient" en PyTorch se implementa con .detach(), que "desconecta"
  un tensor del grafo de calculo, tratandolo como constante en el backward pass.
- "Similitud coseno": producto punto de dos vectores normalizados (longitud 1).
  Va de -1 (opuestos) a 1 (identicos en direccion).
- "Temperatura" (en InfoNCE): parametro que controla que tan "afilada" es la
  distribucion de probabilidad sobre los pares positivo/negativos. Temperaturas
  bajas hacen que el modelo sea mas estricto distinguiendo el positivo correcto.

VALIDACION PREVIA (resultados obtenidos con numpy, antes de este archivo):
- SimSiam: vectores identicos -> loss = -1.0 (colapso, minimo posible) [OK]
- BYOL: equivalencia algebraica MSE(normalizado) = 2 - 2*cos_sim confirmada [OK]
- InfoNCE: positivo similar -> loss=0.013; positivo aleatorio -> loss=1.996 [OK]
- Align-Uniform: vistas parecidas -> L_align bajo; puntos dispersos -> L_uniform bajo [OK]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 1. SimSiam: negative cosine similarity con stop-gradient (Chen & He, Ec. 1-4)
# ----------------------------------------------------------------------------

def negative_cosine_similarity(p, z):
    """
    D(p, z) = -(p/|p|) . (z/|z|)
    z se trata como CONSTANTE (stop-gradient) -- el llamador debe pasar
    z ya con .detach() aplicado, o se aplica aqui mismo por seguridad.
    """
    p = F.normalize(p, dim=1, p=2)
    z = F.normalize(z, dim=1, p=2)
    return -(p * z.detach()).sum(dim=1)


def simsiam_loss(p1, z2, p2, z1):
    """
    Perdida simetrizada de SimSiam (Ec. 4 del paper):
        L = 0.5 * D(p1, stopgrad(z2)) + 0.5 * D(p2, stopgrad(z1))

    Args:
        p1, p2: predicciones de la rama online para cada vista
        z1, z2: proyecciones (SIN predictor) de cada vista

    Returns:
        Escalar (promedio sobre el batch). Rango teorico: [-1, 1].
    """
    d1 = negative_cosine_similarity(p1, z2)
    d2 = negative_cosine_similarity(p2, z1)
    return (0.5 * d1 + 0.5 * d2).mean()


# ----------------------------------------------------------------------------
# 2. BYOL: MSE de vectores L2-normalizados (Grill et al., Ec. 1-2)
# ----------------------------------------------------------------------------

def byol_regression_loss(q, z):
    """
    L = || q/|q| - z/|z| ||_2^2
    z (proyeccion de la red TARGET) se trata como constante -- viene ya
    calculada dentro de un bloque torch.no_grad() en el modelo (ver
    ssl_models.BYOL.forward), asi que no necesita .detach() adicional aqui,
    pero se aplica por seguridad ante cualquier uso incorrecto del modelo.
    """
    q = F.normalize(q, dim=1, p=2)
    z = F.normalize(z, dim=1, p=2)
    return 2 - 2 * (q * z.detach()).sum(dim=1)


def byol_loss(q1, target_z2, q2, target_z1):
    """
    Perdida simetrizada de BYOL (Ec. 2 del paper: L_BYOL = L + L_tilde):
        L = D(q1, target_z2) + D(q2, target_z1)

    Returns:
        Escalar (promedio sobre el batch). Rango teorico: [0, 4].
    """
    loss1 = byol_regression_loss(q1, target_z2)
    loss2 = byol_regression_loss(q2, target_z1)
    return (loss1 + loss2).mean()


# ----------------------------------------------------------------------------
# 3. InfoNCE (usado por CPC, van den Oord et al., Ec. 4)
# ----------------------------------------------------------------------------

def infonce_loss(context, targets, temperature=0.1):
    """
    Perdida InfoNCE tal como se usa en CPC: para cada posicion en la secuencia,
    el contexto (salida de la GRU) debe distinguir la representacion FUTURA
    correcta (positivo) entre todas las representaciones del batch (negativos
    = las representaciones de las OTRAS imagenes/posiciones en el mismo batch).

    Args:
        context: (batch, dim) -- predicciones del contexto en un paso k
                 (ya pasadas por el predictor W_k, ver ssl_models.CPC)
        targets: (batch, dim) -- representaciones reales z_{t+k} de cada
                 elemento del batch. targets[i] es el positivo de context[i];
                 targets[j] para j!=i actua como negativo.
        temperature: controla que tan "afilada" es la distribucion (CPC usa
                     un log-bilinear score sin temperatura explicita, pero
                     temperatura=1 recupera esa formulacion exacta; usamos
                     0.1 por defecto siguiendo la convencion de SimCLR/MoCo
                     para mejor comportamiento numerico)

    Returns:
        Escalar: entropia cruzada promedio sobre el batch (cada fila es una
        clasificacion entre 'batch' clases posibles, donde la clase correcta
        es la diagonal, es decir context[i] debe emparejar con targets[i]).
    """
    context = F.normalize(context, dim=1, p=2)
    targets = F.normalize(targets, dim=1, p=2)

    # Matriz de similitud (batch, batch): logits[i,j] = sim(context[i], targets[j])
    logits = torch.matmul(context, targets.T) / temperature

    # La clase correcta para la fila i es la columna i (targets[i] es su positivo)
    labels = torch.arange(context.shape[0], device=context.device)

    return F.cross_entropy(logits, labels)


def cpc_loss(predictions, z, min_context_steps=1):
    """
    Aplica infonce_loss a traves de TODOS los pasos futuros predichos por CPC
    (ver ssl_models.CPC.forward, que devuelve una lista 'predictions', una
    por cada paso futuro k=1..n_future_steps).

    Args:
        predictions: lista de (batch, n_rows, dim), salida de cpc_model.forward()
        z: (batch, n_rows, dim) representaciones reales codificadas (targets)
        min_context_steps: cuantas filas iniciales usar como contexto minimo
                            antes de empezar a predecir (evita predecir con
                            contexto vacio/insuficiente)

    Returns:
        Escalar: promedio de la perdida InfoNCE sobre todos los pasos k y
        todas las posiciones validas de la secuencia.
    """
    batch_size, n_rows, dim = z.shape
    total_loss = 0.0
    count = 0

    for k, pred_k in enumerate(predictions, start=1):
        # Para el paso futuro k, el contexto en la posicion t predice z en t+k
        valid_t_range = range(min_context_steps - 1, n_rows - k)
        for t in valid_t_range:
            context_t = pred_k[:, t, :]      # (batch, dim)
            target_t_plus_k = z[:, t + k, :]  # (batch, dim)
            total_loss += infonce_loss(context_t, target_t_plus_k)
            count += 1

    if count == 0:
        raise ValueError("No hay suficientes filas en la secuencia para calcular "
                          "CPC loss con los n_future_steps configurados. "
                          "Aumenta n_rows o reduce n_future_steps.")

    return total_loss / count


# ----------------------------------------------------------------------------
# 4. Align-Uniform (Wang & Isola, 2020, Ec. en Sec. 3)
# ----------------------------------------------------------------------------

def align_loss(x, y, alpha=2):
    """
    L_align = E[ ||f(x) - f(y)||_2^alpha ]
    x, y: proyecciones de dos vistas aumentadas del MISMO ejemplo.
    """
    x = F.normalize(x, dim=1, p=2)
    y = F.normalize(y, dim=1, p=2)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    """
    L_uniform = log( E[ exp(-t * ||f(x)-f(y)||_2^2) ] ), para pares x,y
    DISTINTOS dentro del mismo batch (no necesariamente relacionados).
    """
    x = F.normalize(x, dim=1, p=2)
    sq_dists = torch.pdist(x, p=2).pow(2)  # distancias entre todos los pares (sin repetir, sin diagonal)
    return sq_dists.mul(-t).exp().mean().log()


def align_uniform_loss(z1, z2, lambda_align=1.0, lambda_uniform=1.0, alpha=2, t=2):
    """
    Perdida combinada de Align-Uniform (Sec. 4 del paper: se optimiza
    L_align + L_uniform directamente, en vez de InfoNCE).

    Args:
        z1, z2: proyecciones de las dos vistas aumentadas (batch, dim)
        lambda_align, lambda_uniform: pesos relativos (el paper muestra que
            la relacion entre ambos importa mas que sus valores absolutos,
            Sec. 5: "as long as the ratio between two weights is not too large")
        alpha: exponente en L_align (por defecto 2, Sec. 3 del paper)
        t: parametro del kernel gaussiano en L_uniform (por defecto 2)

    Returns:
        Escalar: combinacion ponderada de ambas perdidas.
    """
    l_align = align_loss(z1, z2, alpha=alpha)
    # Uniformidad se calcula sobre AMBAS vistas juntas, para cubrir mejor la
    # distribucion completa de representaciones del batch
    combined = torch.cat([z1, z2], dim=0)
    l_uniform = uniform_loss(combined, t=t)
    return lambda_align * l_align + lambda_uniform * l_uniform, l_align, l_uniform


# ----------------------------------------------------------------------------
# Prueba autocontenida (tensores aleatorios, sin necesitar Food-101 ni GPU)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, dim = 8, 16

    print("1. Probando SimSiam loss...")
    p1, z2, p2, z1 = [torch.randn(batch_size, dim) for _ in range(4)]
    loss = simsiam_loss(p1, z2, p2, z1)
    print(f"   Loss (vectores aleatorios): {loss.item():.4f} (rango esperado: [-1, 1])")
    loss_collapse = simsiam_loss(z2, z2, z1, z1)
    print(f"   Loss (colapso, p==z): {loss_collapse.item():.4f} (debe ser -1.0)")
    assert abs(loss_collapse.item() - (-1.0)) < 1e-4, "Fallo: colapso deberia dar -1.0"

    print("\n2. Probando BYOL loss...")
    q1, tz2, q2, tz1 = [torch.randn(batch_size, dim) for _ in range(4)]
    loss = byol_loss(q1, tz2, q2, tz1)
    print(f"   Loss (vectores aleatorios): {loss.item():.4f} (rango esperado: [0, 8], simetrizado)")
    loss_perfect = byol_loss(tz2, tz2, tz1, tz1)
    print(f"   Loss (prediccion perfecta, q==z): {loss_perfect.item():.4f} (debe ser 0.0)")
    assert loss_perfect.item() < 1e-4, "Fallo: prediccion perfecta deberia dar 0.0"

    print("\n3. Probando InfoNCE / CPC loss...")
    anchor = torch.randn(batch_size, dim)
    positive_good = anchor + torch.randn(batch_size, dim) * 0.05
    loss_good = infonce_loss(anchor, positive_good)
    positive_bad = torch.randn(batch_size, dim)
    loss_bad = infonce_loss(anchor, positive_bad)
    print(f"   Loss (positivo parecido): {loss_good.item():.4f}")
    print(f"   Loss (positivo aleatorio): {loss_bad.item():.4f} (debe ser mayor)")
    assert loss_good.item() < loss_bad.item(), "Fallo: positivo parecido deberia dar menor perdida"

    print("\n4. Probando Align-Uniform loss...")
    z1_au = torch.randn(batch_size, dim)
    z2_au_good = z1_au + torch.randn(batch_size, dim) * 0.05
    combined_loss, l_align, l_uniform = align_uniform_loss(z1_au, z2_au_good)
    print(f"   Combined: {combined_loss.item():.4f}, L_align: {l_align.item():.4f}, "
          f"L_uniform: {l_uniform.item():.4f}")

    print("\n5. Probando CPC loss completo (con secuencia simulada)...")
    n_rows, n_future = 6, 2
    z_seq = torch.randn(batch_size, n_rows, dim)
    predictions_seq = [torch.randn(batch_size, n_rows, dim) for _ in range(n_future)]
    cpc_l = cpc_loss(predictions_seq, z_seq)
    print(f"   CPC loss (secuencia simulada, {n_rows} filas, {n_future} pasos futuros): "
          f"{cpc_l.item():.4f}")

    print("\nssl_losses.py: todas las pruebas OK")
