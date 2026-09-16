# Reducción de Datos en Aprendizaje Auto-Supervisado (SSL) para Visión por Computador

Este repositorio investiga el impacto de la reducción de datos mediante selección de subconjuntos informativos usando **SAS (Subsets that maximize Augmentation Similarity)** en modelos de aprendizaje auto-supervisado (SSL) sobre el dataset **Food-101**.

---

## Métodos SSL Implementados
- **SimSiam**: Siamese representations learning con *stop-gradient* y predictor asimétrico.
- **BYOL**: Bootstrap Your Own Latent con red target actualizada vía promedio móvil exponencial (EMA).
- **CPC**: Contrastive Predictive Coding adaptado a imágenes mediante predicción autorregresiva de parches latentes.
- **Align-Uniform**: Optimización directa de cercanía de pares positivos (*alignment*) y dispersión en la hiperesfera (*uniformity*).

---

## Estructura del Repositorio

- `config_final.py`: Configuración unificada para experimentos controlados (modo `debug` y `full`).
- `data_utils.py`: Transformaciones de datos para SSL (Two-View) y evaluación lineal (Single-View), además de cargadores de Food-101.
- `proxy_model.py`: Extractor de representaciones latentes para la selección de subconjuntos.
- `sas_selection_fast.py`: Algoritmo SAS acelerado en GPU con K-Means para clases latentes no supervisadas.
- `generate_subsets_fast.py`: Generador de subconjuntos con presupuestos exactos (90%, 80%, 60%, 40%) en modos `unsupervised` y `oracle`.
- `ssl_models.py`: Arquitecturas y backbones compartidos (ResNet-18).
- `ssl_losses.py`: Funciones de pérdida simétricas para los 4 métodos SSL.
- `train_ssl_final.py`: Pipeline de entrenamiento SSL controlado con soporte para precisión mixta (AMP).
- `eval_linear.py`: Protocolo de Evaluación Lineal (*Linear Probing*) con congelamiento de backbone y reporte de métricas Top-1 / Top-5.

---

## Instalación

```bash
git clone <URL_DEL_REPOSITORIO>
cd PAE
pip install -r requirements.txt
```

---

## Flujo de Trabajo

### 1. Generación de Subconjuntos SAS (Puro SSL / Sin Etiquetas)
```bash
python generate_subsets_fast.py --mode full --selection_mode unsupervised
```

### 2. Entrenamiento Auto-Supervisado
```bash
# Ejemplo: Entrenar SimSiam con el subconjunto SAS del 60%
python train_ssl_final.py --method simsiam --data_config sas_keep_60pct

# Ejemplo: Entrenar SimSiam con el baseline aleatorio del 60%
python train_ssl_final.py --method simsiam --data_config random_keep_60pct

# Ejemplo: Entrenar SimSiam con el conjunto completo (100%)
python train_ssl_final.py --method simsiam --data_config full
```

### 3. Evaluación Lineal (Linear Probing)
```bash
# Evaluar un modelo específico:
python eval_linear.py --method simsiam --data_config sas_keep_60pct --mode full

# Evaluar todos los checkpoints entrenados en lote:
python eval_linear.py --all --mode full
```
