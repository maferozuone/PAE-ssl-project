# 📊 Matriz Ampliada de Experimentos SSL - PAE (28 Experimentos)

Este documento registra la asignación, cobertura y estado en tiempo real de los **28 experimentos** (4 métodos $\times$ 7 configuraciones de datos) para comparar rigurosamente el comportamiento de **SAS vs. Muestreo Aleatorio** frente al conjunto completo (100\%) a tres tasas de retención: **90\%**, **80\%** y **60\%**.

---

## 🖥️ Matriz Experimental Ampliada (7 Configuraciones)

1. **`full` (100\%)**: 75\,750 imágenes (cota superior de datos)
2. **`sas_keep_90pct` (90\%)**: 68\,175 imágenes (poda leve inteligente del 10\%)
3. **`random_keep_90pct` (90\%)**: 68\,175 imágenes (poda leve estocástica del 10\%)
4. **`sas_keep_80pct` (80\%)**: 60\,600 imágenes (poda moderada inteligente del 20\%)
5. **`random_keep_80pct` (80\%)**: 60\,600 imágenes (poda moderada estocástica del 20\%)
6. **`sas_keep_60pct` (60\%)**: 45\,450 imágenes (poda profunda inteligente del 40\%)
7. **`random_keep_60pct` (60\%)**: 45\,450 imágenes (poda profunda estocástica del 40\%)

---

## 📋 Tablero de Estado de los 28 Experimentos

### 🟢 Tanda 1: PC de la Universidad (`cpc` & `align_uniform`) — RTX A2000

| # | Método | Configuración | Muestras | Estado | Top-1 Acc | Top-5 Acc | Checkpoint |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `align_uniform` | `full` (100\%) | 75\,750 | ✅ **Listo** | **56.61\%** | **81.30\%** | `align_uniform_full_final.pt` |
| 2 | `align_uniform` | `sas_keep_90pct` | 68\,175 | ⏳ **Pendiente** | -- | -- | `align_uniform_sas_keep_90pct_final.pt` |
| 3 | `align_uniform` | `random_keep_90pct` | 68\,175 | ⏳ **Pendiente** | -- | -- | `align_uniform_random_keep_90pct_final.pt` |
| 4 | `align_uniform` | `sas_keep_80pct` | 60\,600 | ✅ **Listo** | **54.30\%** | **79.57\%** | `align_uniform_sas_keep_80pct_final.pt` |
| 5 | `align_uniform` | `random_keep_80pct` | 60\,600 | ⏳ **Pendiente** | -- | -- | `align_uniform_random_keep_80pct_final.pt` |
| 6 | `align_uniform` | `sas_keep_60pct` | 45\,450 | ✅ **Listo** | **51.18\%** | **77.33\%** | `align_uniform_sas_keep_60pct_final.pt` |
| 7 | `align_uniform` | `random_keep_60pct` | 45\,450 | ✅ **Listo** | **51.63\%** | **77.70\%** | `align_uniform_random_keep_60pct_final.pt` |
| 8 | `cpc` | `full` (100\%) | 75\,750 | ✅ **Listo** | **28.34\%** | **55.42\%** | `cpc_full_final.pt` |
| 9 | `cpc` | `sas_keep_90pct` | 68\,175 | ⏳ **Pendiente** | -- | -- | `cpc_sas_keep_90pct_final.pt` |
| 10 | `cpc` | `random_keep_90pct` | 68\,175 | ⏳ **Pendiente** | -- | -- | `cpc_random_keep_90pct_final.pt` |
| 11 | `cpc` | `sas_keep_80pct` | 60\,600 | ✅ **Listo** | **29.19\%** | **56.26\%** | `cpc_sas_keep_80pct_final.pt` |
| 12 | `cpc` | `random_keep_80pct` | 60\,600 | ⏳ **Pendiente** | -- | -- | `cpc_random_keep_80pct_final.pt` |
| 13 | `cpc` | `sas_keep_60pct` | 45\,450 | ✅ **Listo** | **28.74\%** | **55.42\%** | `cpc_sas_keep_60pct_final.pt` |
| 14 | `cpc` | `random_keep_60pct` | 45\,450 | ✅ **Listo** | **29.24\%** | **56.55\%** | `cpc_random_keep_60pct_final.pt` |

> **Comando en el PC de la Universidad para entrenar SOLO los 6 pendientes:**
> ```cmd
> python run_experiments.py --methods cpc align_uniform --data_configs sas_keep_90pct random_keep_90pct random_keep_80pct
> ```
> *(O simplemente `python run_experiments.py --methods cpc align_uniform`, ya que el script detecta y omite automáticamente los que ya están listos).*

---

### 🔵 Tanda 2: Kaggle / GPU Remota (`simsiam` & `byol`)

| # | Método | Configuración | Muestras | Estado | Checkpoint |
| :-: | :--- | :--- | :---: | :---: | :--- |
| 15 | `simsiam` | `full` (100\%) | 75\,750 | ⏳ Pendiente | `simsiam_full_final.pt` |
| 16 | `simsiam` | `sas_keep_90pct` | 68\,175 | ⏳ Pendiente | `simsiam_sas_keep_90pct_final.pt` |
| 17 | `simsiam` | `random_keep_90pct` | 68\,175 | ⏳ Pendiente | `simsiam_random_keep_90pct_final.pt` |
| 18 | `simsiam` | `sas_keep_80pct` | 60\,600 | ⏳ Pendiente | `simsiam_sas_keep_80pct_final.pt` |
| 19 | `simsiam` | `random_keep_80pct` | 60\,600 | ⏳ Pendiente | `simsiam_random_keep_80pct_final.pt` |
| 20 | `simsiam` | `sas_keep_60pct` | 45\,450 | ⏳ Pendiente | `simsiam_sas_keep_60pct_final.pt` |
| 21 | `simsiam` | `random_keep_60pct` | 45\,450 | ⏳ Pendiente | `simsiam_random_keep_60pct_final.pt` |
| 22 | `byol` | `full` (100\%) | 75\,750 | ⏳ Pendiente | `byol_full_final.pt` |
| 23 | `byol` | `sas_keep_90pct` | 68\,175 | ⏳ Pendiente | `byol_sas_keep_90pct_final.pt` |
| 24 | `byol` | `random_keep_90pct` | 68\,175 | ⏳ Pendiente | `byol_random_keep_90pct_final.pt` |
| 25 | `byol` | `sas_keep_80pct` | 60\,600 | ⏳ Pendiente | `byol_sas_keep_80pct_final.pt` |
| 26 | `byol` | `random_keep_80pct` | 60\,600 | ⏳ Pendiente | `byol_random_keep_80pct_final.pt` |
| 27 | `byol` | `sas_keep_60pct` | 45\,450 | ⏳ Pendiente | `byol_sas_keep_60pct_final.pt` |
| 28 | `byol` | `random_keep_60pct` | 45\,450 | ⏳ Pendiente | `byol_random_keep_60pct_final.pt` |

---

## ⚡ Monitor de Progreso Local

Ejecuta en cualquier momento:
```cmd
python check_progress.py
```
