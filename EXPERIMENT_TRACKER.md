# 📊 Matriz de Experimentos SSL - PAE (Seguimiento y Control)

Este documento registra la asignación de experimentos entre **Kaggle** y el **PC de la Universidad (RTX A2000)** para evitar colisiones y asegurar la cobertura de todas las combinaciones de Métodos SSL y Subconjuntos SAS.

---

## 🖥️ Asignación de Recursos

| Máquina | Hardware | Métodos Asignados | Subconjuntos de Datos | Total Runs |
| :--- | :--- | :--- | :--- | :---: |
| **Kaggle** | GPU T4 / P100 (16 GB VRAM) | **`simsiam`**, **`byol`** | `full`, `sas_80%`, `sas_60%`, `random_60%` | **8** |
| **PC de la U** | NVIDIA RTX A2000 (12 GB VRAM) + NVMe 1 TB | **`cpc`**, **`align_uniform`** | `full`, `sas_80%`, `sas_60%`, `random_60%` | **8** |
| **Total Global** | | **4 Métodos** | **4 Configuraciones** | **16 Experimentos** |

---

## 📋 Tablero de Estado de Experimentos

### 🔵 Tanda 1: Kaggle (`simsiam` & `byol`)

| # | Método | Configuración de Datos | Estado | Checkpoint Generado | Tiempo Estimado |
| :-: | :--- | :--- | :---: | :--- | :---: |
| 1 | `simsiam` | `full` (100%) | [ ] Pendiente | `output/checkpoints_final/simsiam_full_final.pt` | ~1.5 h |
| 2 | `simsiam` | `sas_keep_80pct` | [ ] Pendiente | `output/checkpoints_final/simsiam_sas_keep_80pct_final.pt` | ~1.2 h |
| 3 | `simsiam` | `sas_keep_60pct` | [ ] Pendiente | `output/checkpoints_final/simsiam_sas_keep_60pct_final.pt` | ~0.9 h |
| 4 | `simsiam` | `random_keep_60pct` | [ ] Pendiente | `output/checkpoints_final/simsiam_random_keep_60pct_final.pt` | ~0.9 h |
| 5 | `byol` | `full` (100%) | [ ] Pendiente | `output/checkpoints_final/byol_full_final.pt` | ~2.0 h |
| 6 | `byol` | `sas_keep_80pct` | [ ] Pendiente | `output/checkpoints_final/byol_sas_keep_80pct_final.pt` | ~1.6 h |
| 7 | `byol` | `sas_keep_60pct` | [ ] Pendiente | `output/checkpoints_final/byol_sas_keep_60pct_final.pt` | ~1.2 h |
| 8 | `byol` | `random_keep_60pct` | [ ] Pendiente | `output/checkpoints_final/byol_random_keep_60pct_final.pt` | ~1.2 h |

> **Comando de ejecución en Kaggle:** Celda 4 del notebook `kaggle_train.ipynb`.

---

### 🟢 Tanda 2: PC de la Universidad (`cpc` & `align_uniform`)

| # | Método | Configuración de Datos | Estado | Checkpoint Generado | Tiempo Estimado |
| :-: | :--- | :--- | :---: | :--- | :---: |
| 9 | `cpc` | `full` (100%) | [ ] Pendiente | `output/checkpoints_final/cpc_full_final.pt` | ~1.5 h |
| 10 | `cpc` | `sas_keep_80pct` | [ ] Pendiente | `output/checkpoints_final/cpc_sas_keep_80pct_final.pt` | ~1.2 h |
| 11 | `cpc` | `sas_keep_60pct` | [ ] Pendiente | `output/checkpoints_final/cpc_sas_keep_60pct_final.pt` | ~0.9 h |
| 12 | `cpc` | `random_keep_60pct` | [ ] Pendiente | `output/checkpoints_final/cpc_random_keep_60pct_final.pt` | ~0.9 h |
| 13 | `align_uniform` | `full` (100%) | [ ] Pendiente | `output/checkpoints_final/align_uniform_full_final.pt` | ~1.3 h |
| 14 | `align_uniform` | `sas_keep_80pct` | [ ] Pendiente | `output/checkpoints_final/align_uniform_sas_keep_80pct_final.pt` | ~1.0 h |
| 15 | `align_uniform` | `sas_keep_60pct` | [ ] Pendiente | `output/checkpoints_final/align_uniform_sas_keep_60pct_final.pt` | ~0.8 h |
| 16 | `align_uniform` | `random_keep_60pct` | [ ] Pendiente | `output/checkpoints_final/align_uniform_random_keep_60pct_final.pt` | ~0.8 h |

> **Comando de ejecución en el PC de la U:**
> ```cmd
> python run_experiments.py --methods cpc align_uniform
> ```

---

## ⚡ Verificador Automático de Progreso

Puedes ejecutar este script en cualquier momento en tu terminal para ver qué checkpoints ya se completaron:
```cmd
python check_progress.py
```

---

## 📦 Consolidación Final y Evaluación

Una vez completadas ambas tandas:
1. Copiar los archivos `.pt` descargados de Kaggle a la carpeta `output/checkpoints_final/` del PC.
2. Ejecutar la evaluación lineal en el PC para obtener las precisiones Top-1 y Top-5 de todos los modelos:
   ```cmd
   python eval_linear.py --all --mode full
   ```
