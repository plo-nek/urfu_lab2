"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
     # 1. Загрузка датасета
    df = pd.read_csv(data.csv_path)

    # 2. Количество строк
    row_count = int(df.shape[0])

    # 3. Пропуски по столбцам
    missing_by_column = df.isna().sum().to_dict()

    # 4. Пассажиры старше 30 лет (NaN автоматически не учитываются)
    adults_over_30_count = int((df["Age"] > 30).sum())

    # 5. Средний возраст по классам (NaN исключаются)
    mean_age_by_pclass = {
        int(k): float(v)
        for k, v in df.groupby("Pclass")["Age"].mean().items()
    }

    # 6. Доля выживших по классам
    survival_rate_by_pclass = {
        int(k): float(v)
        for k, v in df.groupby("Pclass")["Survived"].mean().items()
    }

    # 7. Пять наибольших тарифов по убыванию
    highest_fares = (
        df["Fare"]
        .sort_values(ascending=False)
        .head(5)
        .tolist()
    )

    return TitanicSummary(
        row_count=row_count,
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )


if __name__ == "__main__":
    result = analyze_titanic(TitanicInput(csv_path="titanic.csv"))

    print("=== Строк в датасете ===")
    print(result.row_count)

    print("\n=== Пропуски по столбцам ===")
    for col, cnt in result.missing_by_column.items():
        print(f"{col}: {cnt}")

    print("\n=== Пассажиров старше 30 ===")
    print(result.adults_over_30_count)

    print("\n=== Средний возраст по классам ===")
    for k in sorted(result.mean_age_by_pclass):
        print(f"Pclass {k}: {result.mean_age_by_pclass[k]:.4f}")

    print("\n=== Доля выживших по классам ===")
    for k in sorted(result.survival_rate_by_pclass):
        print(f"Pclass {k}: {result.survival_rate_by_pclass[k]:.4f}")

    print("\n=== Топ-5 Fare ===")
    print(result.highest_fares)