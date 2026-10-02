"""generate_paper_figures.py
Genera figuras de alta calidad para el articulo IEEEtran a partir de los
archivos JSON de evaluacion lineal en output/results_final/.
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Configuracion de estilo para publicacion cientifica IEEE
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "figure.titlesize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "lines.linewidth": 1.75,
    "lines.markersize": 6,
})

RESULTS_DIR = "output/results_final"
FIG_DIR = "latex/figures"
os.makedirs(FIG_DIR, exist_ok=True)

METHODS = [
    ("align_uniform", "Align-Uniform", "#2ca02c"),   # verde
    ("byol", "BYOL", "#1f77b4"),                     # azul
    ("cpc", "CPC", "#ff7f0e"),                       # naranja
    ("simsiam", "SimSiam", "#d62728"),               # rojo
]

BASE_TIMES_MIN = {
    "align_uniform": 281.7,
    "byol": 305.5,
    "cpc": 184.0,
    "simsiam": 273.8,
}

def load_data():
    data = {}
    for f in glob.glob(os.path.join(RESULTS_DIR, "eval_*.json")):
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        if d.get("mode") == "full":
            m = d.get("method")
            c = d.get("data_config")
            data[(m, c)] = d
    return data

def plot_figure_1_retention_curves(data):
    """Figura 1: Curvas de precision Top-1 vs Porcentaje de datos conservados."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    axes = axes.flatten()

    retention_pcts = [60, 80, 90]

    for idx, (m_key, m_name, color) in enumerate(METHODS):
        ax = axes[idx]
        full_top1 = data[(m_key, "full")]["best_top1"]

        # Valores SAS
        sas_vals = [
            data[(m_key, f"sas_keep_{p}pct")]["best_top1"]
            for p in retention_pcts
        ] + [full_top1]

        # Valores Aleatorio
        rand_vals = [
            data[(m_key, f"random_keep_{p}pct")]["best_top1"]
            for p in retention_pcts
        ] + [full_top1]

        all_pcts = retention_pcts + [100]

        # Linea de referencia modelo completo
        ax.axhline(full_top1, color="#555555", linestyle=":", linewidth=1.2,
                   label=f"Completo 100% ({full_top1:.2f}%)")

        # Curva SAS
        ax.plot(all_pcts, sas_vals, marker="o", color=color, linestyle="-",
                label="SAS (Propuesto)", zorder=4)

        # Curva Aleatorio
        ax.plot(all_pcts, rand_vals, marker="s", color="#888888", linestyle="--",
                label="Aleatorio (Control)", zorder=3)

        ax.set_title(f"({chr(97 + idx)}) {m_name}", fontweight="bold")
        ax.set_xticks(all_pcts)
        ax.set_xticklabels(["60%", "80%", "90%", "100%"])
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.legend(loc="lower right" if m_key != "simsiam" else "center right", framealpha=0.9)

    for i in [2, 3]:
        axes[i].set_xlabel("Datos de Entrenamiento Conservados (%)")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig1_retention_curves.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("-> Guardada Figura 1: fig1_retention_curves")

def plot_figure_2_efficiency_tradeoff(data):
    """Figura 2: Frontera de Pareto y Trade-off Top-1 vs Tiempo de GPU (Horas)."""
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    markers = {"full": "*", "sas": "o", "random": "s"}

    for m_key, m_name, color in METHODS:
        full_top1 = data[(m_key, "full")]["best_top1"]
        base_time_h = BASE_TIMES_MIN[m_key] / 60.0

        # Punto completo
        ax.scatter(base_time_h, full_top1, color=color, marker=markers["full"],
                   s=130, zorder=5, edgecolors="black", linewidths=0.8,
                   label=f"{m_name} (100%)")

        # Puntos SAS y Aleatorio
        for p, frac in [(90, 0.9), (80, 0.8), (60, 0.6)]:
            t_h = base_time_h * frac
            sas_top1 = data[(m_key, f"sas_keep_{p}pct")]["best_top1"]
            rand_top1 = data[(m_key, f"random_keep_{p}pct")]["best_top1"]

            ax.scatter(t_h, sas_top1, color=color, marker=markers["sas"],
                       s=65, zorder=4, alpha=0.9)
            ax.scatter(t_h, rand_top1, color="#999999", marker=markers["random"],
                       s=45, zorder=3, alpha=0.7)

            # Etiqueta suave para SAS
            if p in (80, 90) and m_key in ("align_uniform", "byol"):
                ax.annotate(f"{m_name} SAS {p}%", (t_h, sas_top1),
                            textcoords="offset points", xytext=(-10, 6),
                            fontsize=7.5, color=color, weight="bold")

        # Conectar puntos SAS
        sas_times = [base_time_h * f for f in [0.6, 0.8, 0.9, 1.0]]
        sas_accs = [data[(m_key, f"sas_keep_{p}pct")]["best_top1"] for p in [60, 80, 90]] + [full_top1]
        ax.plot(sas_times, sas_accs, color=color, linestyle="-", alpha=0.6, linewidth=1.2)

    ax.set_xlabel("Tiempo Estimado de Preentrenamiento en GPU RTX A2000 (Horas)")
    ax.set_ylabel("Exactitud de Evaluación Lineal Top-1 (%)")
    ax.set_title("Compromiso entre Eficiencia Computacional y Rendimiento de Representación", fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.95, ncol=2)

    # Anotacion de zona optima
    ax.axvspan(3.6, 4.6, color="#2ca02c", alpha=0.08, zorder=1)
    ax.text(3.65, 33, "Región Óptima de Eficiencia\n(SAS 80% - 90%)",
            fontsize=8, color="#1b6e1b", style="italic", weight="bold")

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig2_efficiency_pareto.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("-> Guardada Figura 2: fig2_efficiency_pareto")

def plot_figure_3_convergence(data):
    """Figura 3: Dinámica de convergencia durante las 100 épocas de evaluación lineal."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)

    configs_to_plot = [
        ("full", "Completo (100%)", "#000000", "-"),
        ("sas_keep_90pct", "SAS 90%", "#1f77b4", "-"),
        ("sas_keep_80pct", "SAS 80%", "#2ca02c", "-"),
        ("sas_keep_60pct", "SAS 60%", "#ff7f0e", "-"),
        ("random_keep_60pct", "Aleatorio 60%", "#999999", "--"),
    ]

    for ax, (m_key, m_name) in zip([ax1, ax2], [("align_uniform", "Align-Uniform"), ("byol", "BYOL")]):
        for cfg_key, label, color, ls in configs_to_plot:
            hist = data[(m_key, cfg_key)].get("history", [])
            epochs = [h["epoch"] for h in hist]
            test_acc = [h["test_top1"] for h in hist]
            ax.plot(epochs, test_acc, label=label, color=color, linestyle=ls,
                    linewidth=1.4 if "full" in cfg_key or "90" in cfg_key else 1.1)

        ax.set_title(f"Convergencia Lineal: {m_name}", fontweight="bold")
        ax.set_xlabel("Época de Linear Probing")
        ax.set_ylabel("Top-1 Test Accuracy (%)")
        ax.set_xlim(1, 100)
        ax.legend(loc="lower right", framealpha=0.9, fontsize=7.5)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig3_convergence.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("-> Guardada Figura 3: fig3_convergence")

def plot_figure_4_retention_bars(data):
    """Figura 4: Tasa de retención relativa (%) frente al modelo 100%."""
    fig, ax = plt.subplots(figsize=(7.0, 3.8))

    budgets = [90, 80, 60]
    n_methods = len(METHODS)
    x = np.arange(len(budgets))
    width = 0.18

    # Colores por metodo
    palette_sas = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    palette_rand = ["#a1d99b", "#9ecae1", "#fdd0a2", "#fcae91"]

    for i, (m_key, m_name, _) in enumerate(METHODS):
        full_top1 = data[(m_key, "full")]["best_top1"]
        sas_ret = [
            (data[(m_key, f"sas_keep_{b}pct")]["best_top1"] / full_top1) * 100
            for b in budgets
        ]
        rand_ret = [
            (data[(m_key, f"random_keep_{b}pct")]["best_top1"] / full_top1) * 100
            for b in budgets
        ]

        offset = (i - 1.5) * width
        ax.bar(x + offset, sas_ret, width=width*0.48, color=palette_sas[i],
               label=f"{m_name} (SAS)" if i == 0 else f"{m_name} (SAS)", edgecolor="black", linewidth=0.5)
        ax.bar(x + offset + width*0.48, rand_ret, width=width*0.48, color=palette_rand[i],
               label=f"{m_name} (Aleatorio)" if i == 0 else f"{m_name} (Aleatorio)", edgecolor="gray", linewidth=0.5, hatch="//")

    ax.axhline(100.0, color="#333333", linestyle="--", linewidth=1.1, label="Paridad 100% (Modelo Completo)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Presupuesto 90%", "Presupuesto 80%", "Presupuesto 60%"])
    ax.set_ylabel("Retención Relativa Top-1 (%)")
    ax.set_title("Comparación de Retención de Rendimiento Relativo: SAS vs. Aleatorio", fontweight="bold")
    ax.set_ylim(85, 142)  # rango ajustado para visualizar claramente incluyendo valores superiores al 100%
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.9, fontsize=7.5)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig4_relative_retention.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("-> Guardada Figura 4: fig4_relative_retention")

def main():
    print("Cargando resultados experimentales...")
    data = load_data()
    print(f"Cargadas {len(data)} combinaciones (metodo, configuracion).")

    plot_figure_1_retention_curves(data)
    plot_figure_2_efficiency_tradeoff(data)
    plot_figure_3_convergence(data)
    plot_figure_4_retention_bars(data)
    print("\n[OK] Todas las figuras generadas exitosamente en 'latex/figures/'.")

if __name__ == "__main__":
    main()
