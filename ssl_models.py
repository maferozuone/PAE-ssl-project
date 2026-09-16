"""
ssl_models.py
Arquitecturas de los 4 metodos de aprendizaje auto-supervisado (SSL) a comparar:
  - SimSiam  (Chen & He, 2021)
  - BYOL     (Grill et al., 2020)
  - CPC      (van den Oord et al., 2018) -- version adaptada a imagenes estaticas
  - Align-Uniform (Wang & Isola, 2020) -- usa la misma arquitectura base que
    SimCLR/SimSiam, pero optimiza L_align + L_uniform en vez de InfoNCE directo

TERMINOS CLAVE (recordatorio):
- "Backbone": la parte de la red que extrae caracteristicas visuales (ResNet
  sin su capa de clasificacion). Es COMPARTIDO por los 4 metodos aqui, para
  que la comparacion entre ellos sea justa (mismo punto de partida arquitectonico).
- "Projector" (proyector): pequena red (MLP, Multi-Layer Perceptron = red con
  varias capas totalmente conectadas) que transforma el embedding del backbone
  a un espacio de menor dimension, donde se calcula la perdida.
- "Predictor": una MLP adicional que SOLO existe en la rama "online" de BYOL
  y SimSiam. Transforma la proyeccion para intentar predecir la proyeccion de
  la OTRA vista/rama.
- "Red target" / "momentum encoder": copia de la red online cuyos pesos se
  actualizan como un promedio movil exponencial (EMA) de los pesos online,
  en vez de por gradiente directo. La usa BYOL (y opcionalmente CPC si se
  quiere estabilidad extra), pero NO SimSiam ni Align-Uniform en su forma base.
- "Stop-gradient": operacion que trata un tensor como una CONSTANTE durante
  el backward pass, es decir, el gradiente no fluye a traves de el. Critica
  para evitar el colapso en SimSiam y BYOL (ver Sec. 4.1 del paper SimSiam).

NOTA SOBRE FIDELIDAD A LOS PAPERS:
- El proyector de SimSiam tiene 3 capas (siguiendo el paper original, Sec. 3),
  con BatchNorm en todas las capas del proyector excepto ReLU en la ultima.
- El predictor de SimSiam/BYOL tiene 2 capas, estructura "bottleneck"
  (dimension de entrada/salida = 2048 en el paper original; en nuestro modo
  debug usamos dimensiones mas chicas via cfg.projector_hidden_dim/output_dim).
- CPC aqui se adapta de su forma original (secuencial, para audio/texto/RL)
  a una version para IMAGENES ESTATICAS: se sigue el enfoque de "CPC v2"
  (Henaff et al. 2019, citado en el paper SAS), donde la imagen se divide en
  parches y un modelo autoregresivo (aqui simplificado a una GRU sobre una
  secuencia de parches) predice representaciones de parches futuros.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import copy


# ----------------------------------------------------------------------------
# 1. Backbone compartido (igual para los 4 metodos, para comparacion justa)
# ----------------------------------------------------------------------------

def build_shared_backbone(name="resnet18", pretrained=False):
    """
    Construye el backbone (ResNet sin capa de clasificacion) que usaran los
    4 metodos SSL. pretrained=False por defecto: en SSL normalmente se
    entrena el backbone DESDE CERO (a diferencia del proxy_model.py, que
    SI usa pesos preentrenados porque solo necesita similitud aproximada).

    Returns:
        (backbone, embedding_dim)
    """
    if name == "resnet18":
        net = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.IMAGENET1K_V1)
        embedding_dim = 512
    elif name == "resnet50":
        net = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.IMAGENET1K_V2)
        embedding_dim = 2048
    else:
        raise ValueError(f"Backbone '{name}' no soportado.")
    net.fc = nn.Identity()
    return net, embedding_dim


# ----------------------------------------------------------------------------
# 2. Projector y Predictor (MLPs, siguiendo SimSiam Sec. 3 / BYOL Sec. 3.3)
# ----------------------------------------------------------------------------

class ProjectionMLP(nn.Module):
    """
    Proyector de 3 capas (SimSiam Sec. 3: "This MLP has 3 layers", BN en todas
    las capas, sin ReLU en la salida). BYOL usa una version de 2 capas; aqui
    generalizamos con un parametro num_layers para poder usar 2 o 3 segun
    el metodo, manteniendo el mismo estilo.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            is_last = (i == num_layers - 1)
            layers.append(nn.BatchNorm1d(dims[i+1]))
            if not is_last:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PredictionMLP(nn.Module):
    """
    Predictor de 2 capas, estructura "bottleneck" (SimSiam Sec. 3: "d=2048,
    hidden=512"). Solo existe en la rama online de SimSiam y BYOL.
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------------
# 3. SimSiam (Chen & He, 2021)
# ----------------------------------------------------------------------------

class SimSiam(nn.Module):
    """
    Arquitectura: encoder (backbone+projector) compartido entre las 2 vistas,
    + predictor SOLO en la rama que se compara (se aplica a ambas ramas por
    simetria, pero conceptualmente 'predice' la proyeccion de la otra vista).
    NO usa red target, NO usa pares negativos. El stop-gradient se aplica
    en la funcion de perdida (ssl_losses.py), no en el modelo.
    """
    def __init__(self, backbone_name, hidden_dim, output_dim):
        super().__init__()
        self.backbone, embedding_dim = build_shared_backbone(backbone_name, pretrained=False)
        self.projector = ProjectionMLP(embedding_dim, hidden_dim, output_dim, num_layers=3)
        self.predictor = PredictionMLP(output_dim, hidden_dim // 4)

    def forward(self, x1, x2):
        """
        Args:
            x1, x2: dos vistas aumentadas del mismo batch de imagenes
        Returns:
            p1, z2, p2, z1: proyecciones (z) y predicciones (p) de cada rama,
                             tal como se usan en ssl_losses.simsiam_loss
        """
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        return p1, z2, p2, z1


# ----------------------------------------------------------------------------
# 4. BYOL (Grill et al., 2020)
# ----------------------------------------------------------------------------

class BYOL(nn.Module):
    """
    Arquitectura: red ONLINE (backbone+projector+predictor, se entrena por
    gradiente) + red TARGET (backbone+projector, se actualiza por EMA, sin
    gradiente directo). El predictor SOLO existe en la rama online (BYOL
    Sec. 3.1, Figura 2: "this predictor is only applied to the online branch").
    """
    def __init__(self, backbone_name, hidden_dim, output_dim, ema_tau=0.99):
        super().__init__()
        self.ema_tau = ema_tau

        # Red online (se entrena por gradiente)
        self.online_backbone, embedding_dim = build_shared_backbone(backbone_name, pretrained=False)
        self.online_projector = ProjectionMLP(embedding_dim, hidden_dim, output_dim, num_layers=2)
        self.predictor = PredictionMLP(output_dim, hidden_dim // 4)

        # Red target (copia inicial de la online, se actualiza SOLO por EMA)
        self.target_backbone = copy.deepcopy(self.online_backbone)
        self.target_projector = copy.deepcopy(self.online_projector)
        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_network(self):
        """
        Actualiza la red target como EMA de la red online (BYOL Ec. 3.1):
            target = tau * target + (1 - tau) * online
        Se llama DESPUES de cada paso de optimizacion (ver train_ssl.py).
        """
        for online_p, target_p in zip(self.online_backbone.parameters(),
                                        self.target_backbone.parameters()):
            target_p.data = self.ema_tau * target_p.data + (1 - self.ema_tau) * online_p.data
        for online_p, target_p in zip(self.online_projector.parameters(),
                                        self.target_projector.parameters()):
            target_p.data = self.ema_tau * target_p.data + (1 - self.ema_tau) * online_p.data

    def forward(self, x1, x2):
        """
        Returns:
            q1, target_z2, q2, target_z1: predicciones online (q) y
            proyecciones target (z, SIN gradiente) para ssl_losses.byol_loss
        """
        # Rama online
        online_z1 = self.online_projector(self.online_backbone(x1))
        online_z2 = self.online_projector(self.online_backbone(x2))
        q1 = self.predictor(online_z1)
        q2 = self.predictor(online_z2)

        # Rama target (sin gradiente, ya que sus parametros tienen requires_grad=False)
        with torch.no_grad():
            target_z1 = self.target_projector(self.target_backbone(x1))
            target_z2 = self.target_projector(self.target_backbone(x2))

        return q1, target_z2, q2, target_z1


# ----------------------------------------------------------------------------
# 5. CPC adaptado a imagenes estaticas (van den Oord et al., 2018;
#    version de parches siguiendo Henaff et al. 2019 "CPC v2")
# ----------------------------------------------------------------------------

class PatchEncoder(nn.Module):
    """
    Codifica cada parche de la imagen en un vector z_ij usando el backbone
    compartido. Sigue el esquema de CPC para imagenes (Sec. 3.2 del paper CPC
    original, adaptado): la imagen se divide en una grilla de parches, y cada
    parche se codifica INDEPENDIENTEMENTE con el mismo backbone.
    """
    def __init__(self, backbone_name, embedding_dim_out):
        super().__init__()
        self.backbone, embedding_dim = build_shared_backbone(backbone_name, pretrained=False)
        self.proj = nn.Linear(embedding_dim, embedding_dim_out)

    def forward(self, patches):
        """
        Args:
            patches: (batch, n_patches, C, H, W)
        Returns:
            z: (batch, n_patches, embedding_dim_out)
        """
        b, n_patches, c, h, w = patches.shape
        flat = patches.view(b * n_patches, c, h, w)
        feats = self.backbone(flat)
        feats = self.proj(feats)
        return feats.view(b, n_patches, -1)


class CPC(nn.Module):
    """
    CPC adaptado a imagenes: divide la imagen en una grilla de parches,
    codifica cada fila con el PatchEncoder, y usa una GRU (modelo
    autoregresivo, ver CPC Sec. 2.2: "we use a GRU for the autoregressive
    part") para resumir el contexto de las filas superiores y predecir
    (via una transformacion lineal W_k, log-bilinear, ver Ec. 3 del paper CPC)
    las representaciones de las filas inferiores.
    """
    def __init__(self, backbone_name, embedding_dim_out, context_dim, n_future_steps=2):
        super().__init__()
        self.encoder = PatchEncoder(backbone_name, embedding_dim_out)
        self.gru = nn.GRU(input_size=embedding_dim_out, hidden_size=context_dim, batch_first=True)
        self.n_future_steps = n_future_steps
        # Una transformacion lineal W_k por cada paso futuro k (CPC Ec. 3: f_k = exp(z^T W_k c))
        self.predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim_out) for _ in range(n_future_steps)
        ])

    def forward(self, patches):
        """
        Args:
            patches: (batch, n_rows, C, H, W) -- cada 'fila' de parches es un
                     paso de la secuencia para la GRU
        Returns:
            z: (batch, n_rows, embedding_dim_out) representaciones codificadas
            c: (batch, n_rows, context_dim) contexto autoregresivo en cada paso
            predictions: lista de (batch, n_rows, embedding_dim_out), una por
                         cada paso futuro k, con la prediccion W_k @ c_t
        """
        z = self.encoder(patches)          # (batch, n_rows, embedding_dim_out)
        c, _ = self.gru(z)                  # (batch, n_rows, context_dim)
        predictions = [predictor(c) for predictor in self.predictors]
        return z, c, predictions


# ----------------------------------------------------------------------------
# 6. Align-Uniform (Wang & Isola, 2020)
# ----------------------------------------------------------------------------

class AlignUniformModel(nn.Module):
    """
    Arquitectura IDENTICA a un encoder contrastivo estandar (backbone +
    projector, SIN predictor, SIN red target -- ver Sec. 4 del paper
    Align-Uniform: se aplica directamente sobre las salidas normalizadas
    del encoder). La diferencia con SimCLR/CPC esta en la funcion de PERDIDA
    (ssl_losses.py: L_align + L_uniform en vez de InfoNCE), no en la arquitectura.
    """
    def __init__(self, backbone_name, hidden_dim, output_dim):
        super().__init__()
        self.backbone, embedding_dim = build_shared_backbone(backbone_name, pretrained=False)
        self.projector = ProjectionMLP(embedding_dim, hidden_dim, output_dim, num_layers=2)

    def forward(self, x1, x2):
        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        return z1, z2


# ----------------------------------------------------------------------------
# 7. Fabrica de modelos (para instanciar segun cfg.ssl_methods)
# ----------------------------------------------------------------------------

def build_ssl_model(method_name, cfg):
    """
    Instancia el modelo SSL correspondiente segun el nombre del metodo,
    usando las dimensiones definidas en config.py.
    """
    if method_name == "simsiam":
        return SimSiam(cfg.backbone, cfg.projector_hidden_dim, cfg.projector_output_dim)
    elif method_name == "byol":
        return BYOL(cfg.backbone, cfg.projector_hidden_dim, cfg.projector_output_dim, ema_tau=0.99)
    elif method_name == "cpc":
        return CPC(cfg.backbone, cfg.projector_output_dim, context_dim=cfg.projector_hidden_dim)
    elif method_name == "align_uniform":
        return AlignUniformModel(cfg.backbone, cfg.projector_hidden_dim, cfg.projector_output_dim)
    else:
        raise ValueError(f"Metodo SSL '{method_name}' no reconocido. "
                          f"Opciones: simsiam, byol, cpc, align_uniform")


# ----------------------------------------------------------------------------
# Prueba autocontenida con tensores aleatorios (sin necesitar Food-101)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    from config import get_config

    cfg = get_config("debug")
    print(f"Configuracion: backbone={cfg.backbone}, hidden={cfg.projector_hidden_dim}, "
          f"output={cfg.projector_output_dim}\n")

    batch_size = 4
    x1 = torch.randn(batch_size, 3, cfg.image_size, cfg.image_size)
    x2 = torch.randn(batch_size, 3, cfg.image_size, cfg.image_size)

    print("1. Probando SimSiam...")
    simsiam = SimSiam(cfg.backbone, cfg.projector_hidden_dim, cfg.projector_output_dim)
    p1, z2, p2, z1 = simsiam(x1, x2)
    print(f"   p1: {p1.shape}, z2: {z2.shape}, p2: {p2.shape}, z1: {z1.shape}")

    print("\n2. Probando BYOL...")
    byol = BYOL(cfg.backbone, cfg.projector_hidden_dim, cfg.projector_output_dim)
    q1, tz2, q2, tz1 = byol(x1, x2)
    print(f"   q1: {q1.shape}, target_z2: {tz2.shape}, q2: {q2.shape}, target_z1: {tz1.shape}")
    print("   Probando actualizacion EMA de la red target...")
    byol.update_target_network()
    print("   EMA OK")

    print("\n3. Probando CPC (adaptado a imagenes, con grilla de parches simulada)...")
    n_rows, patch_size = 4, 32
    patches = torch.randn(batch_size, n_rows, 3, patch_size, patch_size)
    cpc = CPC(cfg.backbone, cfg.projector_output_dim, context_dim=cfg.projector_hidden_dim, n_future_steps=2)
    z, c, predictions = cpc(patches)
    print(f"   z: {z.shape}, c: {c.shape}, predictions[0]: {predictions[0].shape}, "
          f"num_predictions: {len(predictions)}")

    print("\n4. Probando Align-Uniform...")
    align_model = AlignUniformModel(cfg.backbone, cfg.projector_hidden_dim, cfg.projector_output_dim)
    z1_au, z2_au = align_model(x1, x2)
    print(f"   z1: {z1_au.shape}, z2: {z2_au.shape}")

    print("\n5. Probando build_ssl_model (fabrica) para los 4 metodos...")
    for method in cfg.ssl_methods:
        model = build_ssl_model(method, cfg)
        print(f"   {method}: {type(model).__name__} construido OK")

    print("\nssl_models.py: todas las pruebas OK")
