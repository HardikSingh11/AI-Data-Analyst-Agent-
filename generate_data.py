import pandas as pd
import numpy as np

np.random.seed(42)

# ----------------------------
# CONFIGURATION
# ----------------------------
n_rows = 1200
regions = ["North", "South", "East", "West", "Central"]
channels = ["Online", "Retail", "Distributor"]
products = ["Laptop", "Monitor", "Keyboard", "Mouse", "Headphones", "Printer", "Tablet"]
customers = [f"Customer_{i}" for i in range(1, 201)]

# Generate sequential order IDs
order_ids = np.arange(10001, 10001 + n_rows)

# Generate order dates across 2 years
order_dates = pd.date_range(start="2023-01-01", periods=730, freq="D")
order_dates = np.random.choice(order_dates, n_rows)

# ----------------------------
# BASE BUSINESS DATA
# ----------------------------
df = pd.DataFrame({
    "OrderID": order_ids,
    "OrderDate": order_dates,
    "Region": np.random.choice(regions, n_rows, p=[0.22, 0.25, 0.20, 0.20, 0.13]),
    "Channel": np.random.choice(channels, n_rows, p=[0.6, 0.3, 0.1]),
    "Product": np.random.choice(products, n_rows, p=[0.18, 0.15, 0.15, 0.12, 0.18, 0.10, 0.12]),
    "Quantity": np.random.randint(1, 20, n_rows),
    "UnitPrice": np.random.randint(800, 70000, n_rows)
})

# Compute Sales and basic costs
df["Sales"] = df["Quantity"] * df["UnitPrice"]
df["Cost"] = df["Sales"] * np.random.uniform(0.5, 0.9, n_rows)
df["Profit"] = df["Sales"] - df["Cost"]

# ----------------------------
# Adding seasonal and regional patterns
# ----------------------------
# Higher sales in festive months (Nov–Dec)
festive_mask = df["OrderDate"].dt.month.isin([11, 12])
df.loc[festive_mask, "Sales"] *= 1.2

# Laptops and Tablets sell more in Q2
q2_mask = df["OrderDate"].dt.month.isin([4, 5, 6]) & df["Product"].isin(["Laptop", "Tablet"])
df.loc[q2_mask, "Sales"] *= np.random.uniform(1.3, 1.5, q2_mask.sum())

# Online channel gives better margins
online_mask = df["Channel"] == "Online"
df.loc[online_mask, "Profit"] *= 1.15

# ----------------------------
# Inject purposeful anomalies
# ----------------------------
# Extremely large enterprise orders
big_orders = np.random.choice(df.index, size=8, replace=False)
df.loc[big_orders, "Quantity"] = np.random.randint(100, 200, 8)
df.loc[big_orders, "Sales"] *= np.random.uniform(5.0, 10.0, 8)
df.loc[big_orders, "Profit"] *= np.random.uniform(2.0, 3.0, 8)

# Random data‑entry typos
low_price_errors = np.random.choice(df.index, size=6, replace=False)
df.loc[low_price_errors, "UnitPrice"] = np.random.randint(10, 100, 6)
df.loc[low_price_errors, "Sales"] = df.loc[low_price_errors, "Quantity"] * df.loc[low_price_errors, "UnitPrice"]

# ----------------------------
# Derived business metrics
# ----------------------------
df["Year"] = df["OrderDate"].dt.year
df["Month"] = df["OrderDate"].dt.month
df["Customer"] = np.random.choice(customers, n_rows)

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df.to_csv("business_data.csv", index=False)
print("✅  Generated 'business_data.csv' with", len(df), "rows")
print(df.head())