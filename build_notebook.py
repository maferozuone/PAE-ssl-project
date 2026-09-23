"""Script que regenera kaggle_train.ipynb con todos los fixes correctos."""
import json, os

cells = []

# ── CELL 0: Markdown intro ─────────────────────────────────────────────────
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "# PAE - SSL Training Pipeline on Kaggle\n",
        "\n",
        "Repo: https://github.com/maferozuone/PAE-ssl-project\n",
        "\n",
        "**Antes de ejecutar:**\n",
        "- Activa **Internet** en el panel derecho (Session options)\n",
        "- Agrega **Food Images (Food-101)** de K Scott Mader (762 upvotes) via **Add Data**\n",
        "- Selecciona **GPU T4 o P100** en Accelerator\n",
    ]
})

# ── CELL 1: Config ─────────────────────────────────────────────────────────
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "# ============================================================\n",
        "# EDITA ESTOS PARAMETROS ANTES DE EJECUTAR\n",
        "# ============================================================\n",
        "GITHUB_REPO  = 'https://github.com/maferozuone/PAE-ssl-project.git'\n",
        "MODE         = 'full'   # 'debug' para probar rapido\n",
        "SSL_METHODS  = ['simsiam', 'byol']\n",
        "DATA_CONFIGS = ['full', 'sas_keep_60pct', 'sas_keep_80pct', 'random_keep_60pct']\n",
        "N_CLUSTERS   = 101\n",
        "print(f'Modo: {MODE} | Metodos: {SSL_METHODS}')\n",
    ]
})

# ── CELL 2: Clonar repo ────────────────────────────────────────────────────
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 1. Clonar repo e instalar dependencias"]})
cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "import os, sys\n",
        "WORK_DIR = '/kaggle/working/PAE-ssl-project'\n",
        "if not os.path.exists(WORK_DIR):\n",
        "    ret = os.system(f'git clone {GITHUB_REPO} {WORK_DIR}')\n",
        "    if ret != 0: raise RuntimeError('Error al clonar. Verifica que Internet este activado.')\n",
        "    print('Repo clonado!')\n",
        "else:\n",
        "    os.system(f'git -C {WORK_DIR} pull'); print('Repo actualizado.')\n",
        "if WORK_DIR not in sys.path: sys.path.insert(0, WORK_DIR)\n",
        "os.chdir(WORK_DIR)\n",
        "os.system('pip install -q timm')\n",
        "import torch\n",
        "print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')\n",
        "if torch.cuda.is_available(): print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')\n",
    ]
})

# ── CELL 3: Preparar Food-101 ──────────────────────────────────────────────
cells.append({"cell_type": "markdown", "metadata": {}, "source": [
    "## 2. Preparar Food-101\n",
    "\n",
    "Agrega **Food Images (Food-101)** de K Scott Mader (762 upvotes) via **Add Data** en el panel derecho.\n",
]})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "# Detectar dataset — el de K Scott Mader (kmader) tiene:\n",
        "#   imagenes en:  food41/images/\n",
        "#   meta JSONs en: food41/meta/meta/\n",
        "CANDIDATES = [\n",
        "    '/kaggle/input/datasets/kmader/food41',\n",
        "    '/kaggle/input/food41',\n",
        "    '/kaggle/input/food-101/food-101',\n",
        "    '/kaggle/input/food-101',\n",
        "]\n",
        "KAGGLE_FOOD101 = next((p for p in CANDIDATES\n",
        "                       if os.path.exists(p) and os.path.exists(p + '/images')), None)\n",
        "if not KAGGLE_FOOD101:\n",
        "    print('Dataset NO encontrado. Buscando...')\n",
        "    os.system('find /kaggle/input -maxdepth 5 -name images -type d')\n",
        "    raise FileNotFoundError('Agrega Food Images (Food-101) de K Scott Mader desde Add Data')\n",
        "print(f'Dataset encontrado: {KAGGLE_FOOD101}')\n",
        "\n",
        "# Detectar donde estan los JSONs de meta\n",
        "META_OPTIONS = [KAGGLE_FOOD101 + '/meta/meta', KAGGLE_FOOD101 + '/meta']\n",
        "KAGGLE_META  = next((m for m in META_OPTIONS if os.path.exists(m + '/train.json')), None)\n",
        "if not KAGGLE_META:\n",
        "    os.system(f'find {KAGGLE_FOOD101} -name train.json')\n",
        "    raise FileNotFoundError('train.json no encontrado en el dataset')\n",
        "print(f'Meta JSONs en: {KAGGLE_META}')\n",
        "os.system(f'ls {KAGGLE_META}')\n",
    ]
})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "# Crear estructura que torchvision.datasets.Food101 espera:\n",
        "#   data/food-101/images/  (symlink a las imagenes reales ~5 GB)\n",
        "#   data/food-101/meta/    (symlink a la carpeta con train.json y test.json)\n",
        "LOCAL_FOOD101 = os.path.join(WORK_DIR, 'data', 'food-101')\n",
        "os.system(f'rm -rf {LOCAL_FOOD101}')\n",
        "os.makedirs(LOCAL_FOOD101, exist_ok=True)\n",
        "os.symlink(KAGGLE_FOOD101 + '/images', LOCAL_FOOD101 + '/images')\n",
        "os.symlink(KAGGLE_META,                LOCAL_FOOD101 + '/meta')\n",
        "print('Estructura food-101:')\n",
        "os.system(f'ls {LOCAL_FOOD101}/')\n",
        "os.system(f'ls {LOCAL_FOOD101}/meta/')\n",
    ]
})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "import config_final\n",
        "\n",
        "class KaggleConfig(config_final.Config):\n",
        "    def __init__(self, mode='full'):\n",
        "        super().__init__(mode)\n",
        "        self.data_root      = os.path.join(WORK_DIR, 'data')\n",
        "        self.output_root    = os.path.join(WORK_DIR, 'output')\n",
        "        self.checkpoint_dir = os.path.join(self.output_root, 'checkpoints_final')\n",
        "        self.subset_dir     = os.path.join(self.output_root, 'subsets')\n",
        "        self.results_dir    = os.path.join(self.output_root, 'results_final')\n",
        "        for d in [self.data_root, self.output_root, self.checkpoint_dir,\n",
        "                  self.subset_dir, self.results_dir]:\n",
        "            os.makedirs(d, exist_ok=True)\n",
        "        if mode == 'full':\n",
        "            self.num_workers     = 4\n",
        "            self.prefetch_factor = 2\n",
        "\n",
        "config_final.Config     = KaggleConfig\n",
        "config_final.get_config = lambda mode='full': KaggleConfig(mode)\n",
        "cfg = config_final.get_config(MODE)\n",
        "print(f'Config: {cfg}')\n",
        "print(f'  data_root={cfg.data_root} | batch={cfg.batch_size} | epochs={cfg.epochs_ssl} | workers={cfg.num_workers}')\n",
    ]
})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "from data_utils import load_food101\n",
        "print('Cargando Food-101...')\n",
        "dataset = load_food101(cfg, split='train')\n",
        "print(f'Dataset listo: {len(dataset)} muestras')\n",
    ]
})

# ── CELL 4: Embeddings + Subsets ───────────────────────────────────────────
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 3. Embeddings proxy y subsets SAS\n", "\n", "Esta celda puede tardar **15-20 min** en GPU (embeddings ~5-10 min + 4 subsets SAS ~5-10 min).\n"]})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "import numpy as np, time\n",
        "from data_utils import get_eval_transform\n",
        "from proxy_model import ProxyModel, compute_embeddings\n",
        "from sas_selection_fast import approximate_latent_classes, sas_select_from_labels_fast, random_uniform_subset\n",
        "\n",
        "# ---- Embeddings ----\n",
        "emb_path    = os.path.join(cfg.subset_dir, f'embeddings_{cfg.mode}.npy')\n",
        "labels_path = os.path.join(cfg.subset_dir, f'labels_{cfg.mode}.npy')\n",
        "\n",
        "if os.path.exists(emb_path) and os.path.exists(labels_path):\n",
        "    print('Cargando embeddings precalculados...')\n",
        "    embeddings = np.load(emb_path).astype(np.float32)\n",
        "    labels     = np.load(labels_path)\n",
        "else:\n",
        "    print('Calculando embeddings con ResNet18 (ImageNet)... ~5-10 min')\n",
        "    proxy = ProxyModel(cfg.proxy_backbone, pretrained=True)\n",
        "    embeddings, labels = compute_embeddings(\n",
        "        proxy, dataset,\n",
        "        transform=get_eval_transform(cfg.image_size),\n",
        "        device=cfg.device,\n",
        "        batch_size=cfg.batch_size,\n",
        "        num_workers=cfg.num_workers\n",
        "    )\n",
        "    embeddings = embeddings.astype(np.float32)\n",
        "    np.save(emb_path, embeddings)\n",
        "    np.save(labels_path, labels)\n",
        "    print(f'Embeddings guardados: {embeddings.shape}')\n",
        "print(f'Embeddings: {embeddings.shape}')\n",
        "\n",
        "# ---- K-Means ----\n",
        "n_samples = len(dataset)\n",
        "n_clusters = min(N_CLUSTERS, n_samples)\n",
        "latent_path = os.path.join(cfg.subset_dir, f'latent_clusters_k{n_clusters}_{cfg.mode}.npy')\n",
        "if os.path.exists(latent_path):\n",
        "    latent_labels = np.load(latent_path)\n",
        "    print('Clusters K-Means cargados.')\n",
        "else:\n",
        "    print(f'Agrupando en {n_clusters} clusters K-Means...')\n",
        "    latent_labels = approximate_latent_classes(embeddings, n_clusters=n_clusters, seed=42)\n",
        "    np.save(latent_path, latent_labels)\n",
        "    print('Clusters guardados.')\n",
        "\n",
        "# ---- Subsets SAS ----\n",
        "for reduction, keep_frac in zip([10, 20, 40, 60], [0.90, 0.80, 0.60, 0.40]):\n",
        "    target   = round(n_samples * keep_frac)\n",
        "    keep_pct = int(keep_frac * 100)\n",
        "    sas_path  = os.path.join(cfg.subset_dir, f'sas_keep_{keep_pct}pct_{cfg.mode}.npy')\n",
        "    rand_path = os.path.join(cfg.subset_dir, f'random_keep_{keep_pct}pct_{cfg.mode}.npy')\n",
        "    if os.path.exists(sas_path):\n",
        "        print(f'[{keep_pct}%] Ya existe, saltando.')\n",
        "        continue\n",
        "    t0 = time.perf_counter()\n",
        "    result = sas_select_from_labels_fast(\n",
        "        embeddings, latent_labels, total_budget=target,\n",
        "        device=cfg.device, refine=False, verbose=True\n",
        "    )\n",
        "    np.save(sas_path, result['selected_indices'])\n",
        "    np.save(rand_path, random_uniform_subset(n_samples, keep_frac, seed=42))\n",
        "    print(f'  [{keep_pct}%] {len(result[\"selected_indices\"])} muestras [{time.perf_counter()-t0:.1f}s]')\n",
        "\n",
        "print('\\nSubsets disponibles:')\n",
        "os.system(f'ls -lh {cfg.subset_dir}')\n",
    ]
})

# ── CELL 5: Entrenamiento ──────────────────────────────────────────────────
cells.append({"cell_type": "markdown", "metadata": {}, "source": [
    "## 4. Entrenamiento SSL\n",
    "\n",
    "El loop es **reanudable**: si el kernel se cae, vuelve a ejecutar y salta los checkpoints existentes.\n",
]})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "import train_ssl_final as train_module\n",
        "import json, time\n",
        "\n",
        "RESULTS = []\n",
        "TOTAL   = len(SSL_METHODS) * len(DATA_CONFIGS)\n",
        "run_idx = 0\n",
        "print(f'Experimentos: {TOTAL} | Metodos: {SSL_METHODS}')\n",
        "print('=' * 60)\n",
        "\n",
        "for method in SSL_METHODS:\n",
        "    for data_config in DATA_CONFIGS:\n",
        "        run_idx += 1\n",
        "        ckpt = os.path.join(cfg.checkpoint_dir, f'{method}_{data_config}_final.pt')\n",
        "        print(f'\\n[{run_idx}/{TOTAL}] {method.upper()} / {data_config}')\n",
        "        print('-' * 50)\n",
        "        if os.path.exists(ckpt):\n",
        "            print('Checkpoint ya existe, saltando.')\n",
        "            RESULTS.append({'method': method, 'data_config': data_config, 'status': 'skipped'})\n",
        "            continue\n",
        "        t0 = time.perf_counter()\n",
        "        try:\n",
        "            train_module.train(method, data_config, mode=MODE, use_dummy=False)\n",
        "            elapsed = time.perf_counter() - t0\n",
        "            print(f'Completado en {elapsed/60:.1f} min')\n",
        "            RESULTS.append({'method': method, 'data_config': data_config,\n",
        "                            'status': 'done', 'elapsed_min': round(elapsed/60, 1)})\n",
        "        except Exception as e:\n",
        "            print(f'ERROR: {e}')\n",
        "            RESULTS.append({'method': method, 'data_config': data_config,\n",
        "                            'status': 'error', 'error': str(e)})\n",
        "\n",
        "print('\\nRESUMEN:')\n",
        "for r in RESULTS:\n",
        "    icon = {'done': 'OK', 'skipped': '--', 'error': '!!'}[r['status']]\n",
        "    mins = f\" ({r.get('elapsed_min','?')} min)\" if r['status'] == 'done' else ''\n",
        "    print(f\"[{icon}] {r['method']:15s} | {r['data_config']:25s}{mins}\")\n",
    ]
})

# ── CELL 6: Guardar ZIP ────────────────────────────────────────────────────
cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 5. Guardar resultados y descargar\n"]})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": [
        "import zipfile, json\n",
        "\n",
        "summary_path = os.path.join(cfg.results_dir, 'kaggle_run_summary.json')\n",
        "with open(summary_path, 'w') as f:\n",
        "    json.dump(RESULTS, f, indent=2)\n",
        "print(f'Resumen guardado: {summary_path}')\n",
        "\n",
        "output_zip = '/kaggle/working/results_and_checkpoints.zip'\n",
        "output_dir = os.path.join(WORK_DIR, 'output')\n",
        "print('Creando ZIP...')\n",
        "with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zf:\n",
        "    for root, dirs, files in os.walk(output_dir):\n",
        "        for file in files:\n",
        "            fp = os.path.join(root, file)\n",
        "            zf.write(fp, os.path.relpath(fp, output_dir))\n",
        "\n",
        "print(f'ZIP listo: {output_zip} ({os.path.getsize(output_zip)/1e6:.1f} MB)')\n",
        "print('Descargalo desde el panel Output de Kaggle.')\n",
        "os.system(f'ls -lh {cfg.checkpoint_dir}')\n",
    ]
})

# ── Ensamblar notebook ─────────────────────────────────────────────────────
nb = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "language": "python", "sourceType": "notebook"}
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_train.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook generado: {out_path}")
