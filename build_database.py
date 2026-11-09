import sqlite3
import pandas as pd
from common import CLEAN, DB, ensure_dirs

def main():
    ensure_dirs()
    if DB.exists():
        DB.unlink()
    with sqlite3.connect(DB) as con:
        customers = pd.read_csv(CLEAN / "customers.csv")
        products = pd.read_csv(CLEAN / "products.csv")
        orders = pd.read_csv(CLEAN / "orders.csv")
        customers.to_sql("customers", con, index=False, if_exists="replace")
        products.to_sql("products", con, index=False, if_exists="replace")
        orders.to_sql("orders", con, index=False, if_exists="replace")
        con.executescript("""
        CREATE INDEX idx_orders_customer ON orders(customer_id);
        CREATE INDEX idx_orders_product ON orders(product_id);
        CREATE INDEX idx_orders_ts ON orders(order_ts);
        """)
        fk = con.execute("""
            SELECT COUNT(*) FROM orders o
            LEFT JOIN customers c ON o.customer_id=c.customer_id
            LEFT JOIN products p ON o.product_id=p.product_id
            WHERE c.customer_id IS NULL OR p.product_id IS NULL
        """).fetchone()[0]
        print(f"SQLite database: {DB}")
        print("Tables:", [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()])
        print("Unmatched foreign keys:", fk)

if __name__ == "__main__":
    main()
