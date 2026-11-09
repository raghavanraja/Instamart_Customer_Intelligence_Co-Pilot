import pandas as pd
import numpy as np
from common import RAW, CLEAN, ensure_dirs

def clean_customers():
    df = pd.read_csv(RAW / "customers.csv")
    df["city"] = df["city"].astype("string").str.strip().str.title()
    df["gender"] = df["gender"].astype("string").str.strip().str.lower()
    df["gender"] = df["gender"].replace({"m": "male", "f": "female", "": pd.NA})
    df["gender"] = df["gender"].where(df["gender"].isin(["male", "female"]), pd.NA)
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[~df["age"].between(18, 100), "age"] = np.nan
    df["age"] = df["age"].fillna(df["age"].median()).round().astype(int)
    df = df.drop_duplicates(subset=["customer_id"], keep="first")
    return df

def clean_products():
    df = pd.read_csv(RAW / "products.csv")
    df["product_name"] = df["product_name"].astype("string").str.strip()
    df["category"] = df["category"].astype("string").str.strip().str.title()
    df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
    df = df.drop_duplicates(subset=["product_id"], keep="first")
    df = df.dropna(subset=["product_id", "product_name", "category", "price_inr"])
    return df

def clean_orders(customers, products):
    df = pd.read_csv(RAW / "orders.csv")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce")
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")
    df["delivery_minutes"] = pd.to_numeric(df["delivery_minutes"], errors="coerce")
    df["order_ts"] = pd.to_datetime(df["order_ts"], errors="coerce")
    df["status"] = df["status"].astype("string").str.strip().str.lower()
    df = df.drop_duplicates(subset=["order_id"], keep="first")
    df = df.dropna(subset=["order_id", "customer_id", "product_id", "order_ts", "quantity", "order_amount"])
    df = df[df["customer_id"].isin(customers["customer_id"])]
    df = df[df["product_id"].isin(products["product_id"])]
    df = df[(df["quantity"] > 0) & (df["order_amount"] >= 0)]
    df = df[df["status"].isin(["completed", "cancelled", "pending"])]
    # Negative or implausible delivery times are invalid; missing values remain missing.
    df.loc[df["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    df["quantity"] = df["quantity"].astype(int)
    df["discount_pct"] = df["discount_pct"].fillna(0).clip(0, 100)
    df["order_amount"] = df["order_amount"].round(2)
    return df.sort_values("order_ts").reset_index(drop=True)

def main():
    ensure_dirs()
    customers = clean_customers()
    products = clean_products()
    orders = clean_orders(customers, products)
    for name, frame in (("customers", customers), ("products", products), ("orders", orders)):
        frame.to_csv(CLEAN / f"{name}.csv", index=False)
        print(f"{name}: {len(frame)} rows")
    print("Validation: unique customer/product/order IDs:",
          customers.customer_id.is_unique, products.product_id.is_unique, orders.order_id.is_unique)
    print("Validation: all order customer/product keys resolve:",
          orders.customer_id.isin(customers.customer_id).all(),
          orders.product_id.isin(products.product_id).all())

if __name__ == "__main__":
    main()
