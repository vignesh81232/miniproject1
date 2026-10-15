from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


data_path = Path(__file__).resolve().with_name("trades_orderbook_clean.csv")
df = pd.read_csv(data_path)

print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())

df = df.drop_duplicates()
df = df.dropna()

df["side"] = df["side"].str.strip().str.lower()

df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df = df.dropna(subset=["price", "quantity"])
df = df[(df["price"] > 0) & (df["quantity"] > 0)]

df["spread"] = df["ask_price_1"] - df["bid_price_1"]
df["mid_price"] = (df["ask_price_1"] + df["bid_price_1"]) / 2

df["imbalance"] = (
    (df["bid_qty_1"] - df["ask_qty_1"]) /
    (df["bid_qty_1"] + df["ask_qty_1"])
)

df["price"].plot(title="Trade Price")
plt.show()

df["side"].value_counts().plot.bar(title="Buy vs Sell")
plt.show()

features = df[["price", "quantity", "spread", "imbalance"]]

feature_means = features.mean()
feature_scales = features.std(ddof=0).replace(0, 1)
scaled_data = ((features - feature_means) / feature_scales).to_numpy()

print("\nScaled data:\n", scaled_data[:5])

df.to_csv(data_path, index=False)

print("\nData cleaning completed!")
print("Cleaned dataset saved successfully.")