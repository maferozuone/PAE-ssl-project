"""config.py actualizado para experimentos con porcentajes explicitos."""

import os
import torch


class Config:
    def __init__(self, mode="debug"):
        assert mode in ("debug", "full")
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_root = "./data"
        self.output_root = "./output"
        self.checkpoint_dir = os.path.join(self.output_root, "checkpoints")
        self.subset_dir = os.path.join(self.output_root, "subsets")
        self.results_dir = os.path.join(self.output_root, "results")
        for directory in [self.data_root, self.output_root, self.checkpoint_dir,
                          self.subset_dir, self.results_dir]:
            os.makedirs(directory, exist_ok=True)

        self.dataset_name = "food101"
        self.num_classes = 101
        self.image_size = 128

        if mode == "debug":
            self.debug_fraction = 0.02
            self.epochs_ssl = 2
            self.epochs_linear_eval = 3
            self.batch_size = 4
            self.num_workers = 0
            self.backbone = "resnet18"
            self.proxy_backbone = "resnet18"
            self.log_every = 5
            self.use_amp = False
        else:
            self.debug_fraction = 1.0
            self.epochs_ssl = 200
            self.epochs_linear_eval = 100
            self.batch_size = 64
            self.num_workers = 4
            self.backbone = "resnet18"
            self.proxy_backbone = "resnet18"
            self.log_every = 20
            self.use_amp = True

        self.learning_rate = 3e-4
        self.weight_decay = 1e-6
        self.momentum = 0.9

        # Estos son porcentajes CONSERVADOS.
        # Equivalen a reducciones del 10%, 20%, 40% y 60%.
        self.sas_subset_fractions = [0.90, 0.80, 0.60, 0.40]
        self.sas_reduction_percentages = [10, 20, 40, 60]

        self.ssl_methods = ["simsiam", "byol", "cpc", "align_uniform"]
        self.projector_hidden_dim = 2048 if mode == "full" else 256
        self.projector_output_dim = 256 if mode == "full" else 64

    def __repr__(self):
        return f"<Config mode={self.mode} device={self.device} dataset={self.dataset_name}>"


def get_config(mode="debug"):
    return Config(mode=mode)


if __name__ == "__main__":
    debug = get_config("debug")
    full = get_config("full")
    print(debug)
    print(f"  device={debug.device}, batch_size={debug.batch_size}, "
          f"conservado={debug.sas_subset_fractions}")
    print(full)
    print(f"  device={full.device}, batch_size={full.batch_size}, "
          f"conservado={full.sas_subset_fractions}")
