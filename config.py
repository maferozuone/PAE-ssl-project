"""
config.py
Configuracion central del proyecto SAS + SSL sobre Food-101.

Este archivo es el UNICO lugar donde cambias entre modo "debug" (tu PC, sin GPU,
para verificar que el codigo corre) y modo "full" (PC de la universidad, RTX A2000).

Uso:
    from config import get_config
    cfg = get_config(mode="debug")   # o mode="full"

CORRECCION IMPORTANTE (encontrada al probar BYOL con subconjuntos SAS chicos):
El batch_size de modo debug era 32, pero el dataset dummy en modo debug tiene
solo 50 muestras totales, y los subconjuntos SAS/random mas chicos (10%) tienen
apenas 10-12 muestras. Con batch_size=32 y drop_last=True, NINGUN batch
completo cabia en esos subconjuntos, y el DataLoader quedaba vacio (0 batches),
lanzando un error claro (ver train_ssl.py) pero bloqueando la prueba. Se
redujo batch_size a 4 en modo debug, que funciona con todos los subconjuntos
generados (el mas chico tiene 10 muestras -> 2 batches de 4). Ademas, BYOL usa
BatchNorm1d, que requiere batch_size >= 2, asi que 4 tambien cumple ese margen
de seguridad. En modo full, Food-101 real tiene decenas de miles de imagenes
incluso en el subconjunto de 10%, por lo que batch_size=256 nunca tendra este
problema.
"""

import torch
import os


class Config:
    def __init__(self, mode="debug"):
        assert mode in ("debug", "full"), "mode debe ser 'debug' o 'full'"
        self.mode = mode

        # ------------------------------------------------------------------
        # Dispositivo: se detecta automaticamente. En tu PC sera 'cpu'.
        # En la PC de la universidad, si CUDA esta bien instalado, sera 'cuda'.
        # ------------------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------------
        # Rutas de datos y salidas
        # ------------------------------------------------------------------
        self.data_root = "./data"
        self.output_root = "./output"
        self.checkpoint_dir = os.path.join(self.output_root, "checkpoints")
        self.subset_dir = os.path.join(self.output_root, "subsets")
        self.results_dir = os.path.join(self.output_root, "results")

        for d in [self.data_root, self.output_root, self.checkpoint_dir,
                  self.subset_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)

        # ------------------------------------------------------------------
        # Dataset
        # ------------------------------------------------------------------
        self.dataset_name = "food101"
        self.num_classes = 101
        self.image_size = 128        # redimensionamos Food-101 a 128x128 (original es variable, alto res)

        # ------------------------------------------------------------------
        # Parametros que CAMBIAN segun el modo
        # ------------------------------------------------------------------
        if mode == "debug":
            # Modo para tu PC: dataset MUY reducido, pocas epocas, batch chico.
            # El objetivo NO es entrenar un modelo bueno, sino confirmar que
            # el pipeline completo corre sin errores de codigo.
            self.debug_fraction = 0.02      # usamos solo 2% de Food-101 (~2000 imagenes)
            self.epochs_ssl = 2              # solo 2 epocas de SSL
            self.epochs_linear_eval = 3      # solo 3 epocas del clasificador lineal
            self.batch_size = 4              # CORREGIDO: antes era 32, rompia con subsets SAS chicos (10-12 muestras)
            self.num_workers = 0             # 0 evita problemas de multiprocessing en Windows/debug
            self.backbone = "resnet18"
            self.proxy_backbone = "resnet18"
            self.log_every = 5
            self.use_amp = False

        else:  # mode == "full"
            # Modo para la PC de la universidad (RTX A2000, 64GB RAM).
            self.debug_fraction = 1.0        # dataset completo
            self.epochs_ssl = 200
            self.epochs_linear_eval = 100
            self.batch_size = 256
            self.num_workers = 8
            self.backbone = "resnet50"
            self.proxy_backbone = "resnet50"
            self.log_every = 20
            self.use_amp = True              # mixed precision (FP16), aprovecha los Tensor Cores de la A2000

        # ------------------------------------------------------------------
        # Hiperparametros comunes de optimizacion (iguales en ambos modos,
        # siguiendo los papers de referencia: SimCLR/BYOL usan LARS o SGD+momentum,
        # aqui usamos Adam por simplicidad y estabilidad en pocas epocas tambien)
        # ------------------------------------------------------------------
        self.learning_rate = 3e-4
        self.weight_decay = 1e-6
        self.momentum = 0.9

        # ------------------------------------------------------------------
        # Parametros especificos de SAS
        # (ver Seccion 4.5 del paper SAS: subset_fractions, umbral de similitud)
        # ------------------------------------------------------------------
        self.sas_subset_fractions = [0.10, 0.20, 0.40, 0.60]
        self.sas_similarity_threshold = 0.0   # se ajusta empiricamente, ver notas en sas_selection.py

        # ------------------------------------------------------------------
        # Modelos SSL a comparar
        # ------------------------------------------------------------------
        self.ssl_methods = ["simsiam", "byol", "cpc", "align_uniform"]

        # ------------------------------------------------------------------
        # Proyeccion / dimensiones latentes (siguiendo BYOL/SimSiam: 2048->256)
        # ------------------------------------------------------------------
        self.projector_hidden_dim = 2048 if mode == "full" else 256
        self.projector_output_dim = 256 if mode == "full" else 64

    def __repr__(self):
        return f"<Config mode={self.mode} device={self.device} dataset={self.dataset_name}>"


def get_config(mode="debug"):
    return Config(mode=mode)


if __name__ == "__main__":
    # Prueba rapida: ejecuta 'python config.py' para ver que se imprime bien
    cfg_debug = get_config("debug")
    cfg_full = get_config("full")
    print(cfg_debug)
    print(f"  device={cfg_debug.device}, fraction={cfg_debug.debug_fraction}, "
          f"epochs_ssl={cfg_debug.epochs_ssl}, batch_size={cfg_debug.batch_size}")
    print(cfg_full)
    print(f"  device={cfg_full.device}, fraction={cfg_full.debug_fraction}, "
          f"epochs_ssl={cfg_full.epochs_ssl}, batch_size={cfg_full.batch_size}")
