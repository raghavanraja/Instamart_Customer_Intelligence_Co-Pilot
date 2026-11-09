import sqlite3
import pandas as pd
from common import DB, OUTPUTS

def main():
    with sqlite3.connect(DB) as con:
        queries = {
            "city_revenue": """SELECT c.city, COUNT(*) orders, ROUND(SUM(o.order_amount),2) revenue
                FROM orders o JOIN customers c USING(customer_id)
                WHERE o.status='completed' GROUP BY c.city ORDER BY revenue DESC""",
            "category_revenue": """SELECT p.category, COUNT(*) lines, ROUND(SUM(o.order_amount),2) revenue
                FROM orders o JOIN products p USING(product_id)
                WHERE o.status='completed' GROUP BY p.category ORDER BY revenue DESC""",
            "monthly_trend": """SELECT substr(order_ts,1,7) month, COUNT(*) orders,
                ROUND(SUM(order_amount),2) revenue FROM orders
                WHERE status='completed' GROUP BY month ORDER BY month""",
            "status_counts": "SELECT status, COUNT(*) count FROM orders GROUP BY status",
        }
        report = []
        for name, sql in queries.items():
            df = pd.read_sql_query(sql, con)
            df.to_csv(OUTPUTS / f"part_a_{name}.csv", index=False)
            report.append(f"{name}\n{df.to_string(index=False)}")
    (OUTPUTS / "part_a_eda_report.txt").write_text("\n\n".join(report), encoding="utf-8")
    print("Part A EDA reports saved.")

if __name__ == "__main__":
    main()
