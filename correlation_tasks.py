"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])

    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]

    men_count = int(men.shape[0])
    women_count = int(women.shape[0])

    women_corr = women[FEATURES + [TARGET]].corr(method="pearson")[TARGET].drop(TARGET)
    men_corr = men[FEATURES + [TARGET]].corr(method="pearson")[TARGET].drop(TARGET)

    women_mri_correlation = {k: float(v) for k, v in women_corr.items()}
    men_mri_correlation = {k: float(v) for k, v in men_corr.items()}

    best_feature = None
    best_abs = -1.0

    for feature in FEATURES:
        w_val = abs(women_mri_correlation.get(feature, 0.0))
        m_val = abs(men_mri_correlation.get(feature, 0.0))
        
        if w_val > best_abs:
            best_abs = w_val
            best_feature = feature
            
        if m_val > best_abs:
            best_abs = m_val
            best_feature = feature

    return BrainCorrelationSummary(
        men_count=men_count,
        women_count=women_count,
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=best_feature,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    result = analyze_brain_correlations(BrainDataInput(csv_path="brainsize.txt"))

    print("=== Количество наблюдений ===")
    print(f"Men: {result.men_count}")
    print(f"Women: {result.women_count}")

    print("\n=== Корреляции с MRI_Count (Women) ===")
    for k, v in result.women_mri_correlation.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Корреляции с MRI_Count (Men) ===")
    for k, v in result.men_mri_correlation.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Сильнейший признак ===")
    print(result.strongest_mri_feature)

    df = pd.read_csv("brainsize.txt", sep="\t", na_values=["NA"])
    plot_features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    for ax, feature in zip(axes, plot_features):
        sns.scatterplot(
            data=df, x=feature, y="MRI_Count", hue="Gender", ax=ax
        )
        ax.set_title(f"{feature} vs MRI_Count")
    plt.tight_layout()
    plt.savefig("scatter_mri.png", dpi=120)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, gender in zip(axes, ["Female", "Male"]):
        sub = df[df["Gender"] == gender][plot_features + ["MRI_Count"]]
        sns.heatmap(
            sub.corr(method="pearson"),
            annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax,
        )
        ax.set_title(f"Correlation heatmap: {gender}")
    plt.tight_layout()
    plt.savefig("heatmap_mri.png", dpi=120)
