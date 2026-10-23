from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "archive" / "fmcg_sales_3years_1M_rows.csv"


def load_sales_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_daily_demand_table(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("date", as_index=False).agg(
        demand=("units_sold", "sum"),
        revenue=("net_sales", "sum"),
        avg_discount=("discount_pct", "mean"),
        promo_rate=("promo_flag", "mean"),
        stock_out_rate=("stock_out_flag", "mean"),
        avg_temp=("temperature", "mean"),
        avg_rain=("rain_mm", "mean"),
        avg_lead_time=("lead_time_days", "mean"),
        avg_margin=("margin_pct", "mean"),
        holiday_rate=("is_holiday", "mean"),
    )

    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["day_of_month"] = daily["date"].dt.day
    daily["month"] = daily["date"].dt.month
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["is_weekend"] = daily["date"].dt.dayofweek.isin([5, 6]).astype(int)
    daily["year"] = daily["date"].dt.year

    for lag in [1, 7, 14, 30]:
        daily[f"lag_{lag}"] = daily["demand"].shift(lag)
    daily["rolling_mean_7"] = daily["demand"].shift(1).rolling(window=7, min_periods=1).mean()
    daily["rolling_std_7"] = daily["demand"].shift(1).rolling(window=7, min_periods=1).std().fillna(0)

    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def train_forecast_model(daily: pd.DataFrame, test_days: int = 30):
    feature_columns = [
        "day_of_week",
        "day_of_month",
        "month",
        "week_of_year",
        "day_of_year",
        "is_weekend",
        "avg_discount",
        "promo_rate",
        "avg_temp",
        "avg_rain",
        "avg_lead_time",
        "avg_margin",
        "holiday_rate",
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_std_7",
    ]

    model_data = daily[feature_columns + ["demand"]].copy()
    model_data = model_data.dropna().reset_index(drop=True)

    split_index = len(model_data) - test_days
    train = model_data.iloc[:split_index]
    test = model_data.iloc[split_index:]

    X_train = train[feature_columns]
    y_train = train["demand"]
    X_test = test[feature_columns]
    y_test = test["demand"]

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    forecast_df = pd.DataFrame({
        "date": daily["date"].iloc[-test_days:].reset_index(drop=True),
        "actual": y_test.to_numpy(),
        "forecast": predictions,
      })

    return model, forecast_df, {"mae": mae, "rmse": rmse, "r2": r2}


def compute_inventory_policy(daily: pd.DataFrame, service_level_z: float = 1.645) -> dict:
    demand = daily["demand"]
    avg_daily_demand = demand.mean()
    std_daily_demand = demand.std(ddof=1)
    lead_time_days = float(daily["avg_lead_time"].mean())
    safety_stock = service_level_z * std_daily_demand * np.sqrt(lead_time_days)
    reorder_point = avg_daily_demand * lead_time_days + safety_stock

    return {
        "avg_daily_demand": avg_daily_demand,
        "std_daily_demand": std_daily_demand,
        "lead_time_days": lead_time_days,
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
    }


def plot_forecast(forecast_df: pd.DataFrame, metrics: dict, output_path: Path | None = None):
    plt.figure(figsize=(12, 5))
    plt.plot(forecast_df["date"], forecast_df["actual"], label="Actual demand", color="steelblue", linewidth=2)
    plt.plot(forecast_df["date"], forecast_df["forecast"], label="Forecast", color="darkorange", linewidth=2)
    plt.title("Daily Demand Forecast vs Actual")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(8, 4))
    snacks = ["MAE", "RMSE", "R2"]
    values = [metrics["mae"], metrics["rmse"], metrics["r2"]]
    plt.bar(snacks, values, color=["#4C78A8", "#F58518", "#54A24B"])
    plt.ylabel("Value")
    plt.title("Forecast Accuracy")
    plt.grid(True, axis="y", alpha=0.3)
    if output_path is not None:
        plt.savefig(output_path.with_name("forecast_accuracy.png"), bbox_inches="tight")
    plt.show()


def main() -> None:
    df = load_sales_data()
    daily = build_daily_demand_table(df)
    model, forecast_df, metrics = train_forecast_model(daily)
    inventory_policy = compute_inventory_policy(daily)

    print("Dataset rows:", len(df))
    print("Daily demand summary:")
    print(daily["demand"].describe())
    print("\nForecast metrics:")
    print({key: round(float(value), 4) if isinstance(value, (float, np.floating)) else value for key, value in metrics.items()})
    print("\nInventory policy:")
    for key, value in inventory_policy.items():
        print(f"{key}: {value:,.2f}")

    output_dir = ROOT / "data"
    output_dir.mkdir(exist_ok=True)
    daily.to_csv(output_dir / "daily_inventory_demand.csv", index=False)
    forecast_df.to_csv(output_dir / "forecast_demand_results.csv", index=False)

    plot_forecast(forecast_df, metrics, output_dir / "forecast_vs_actual.png")


if __name__ == "__main__":
    main()
