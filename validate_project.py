"""Quick integrity checks for generated capstone deliverables."""
from pathlib import Path
import json, sqlite3
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
required = [
    "data/raw/customers.csv", "data/raw/products.csv", "data/raw/orders.csv",
    "data/raw/tickets.csv", "data/policies/refund_policy.txt",
    "data/clean/customers.csv", "data/clean/products.csv", "data/clean/orders.csv",
    "data/instamart.db", "sql/queries.sql", "outputs/customer_segments.csv",
    "outputs/lapse_model.joblib", "outputs/retention_list.csv",
    "outputs/triage_results.csv", "outputs/triage_budget_demo.csv",
    "outputs/agent_runs.jsonl", "outputs/refund_ledger.jsonl",
    "outputs/policy_eval_results.csv", "outputs/plan_execute_comparison.csv",
    "state/agent_checkpoint.json",
]
missing = [x for x in required if not (ROOT / x).exists()]
assert not missing, f"Missing files: {missing}"
customers = pd.read_csv(ROOT/"data/clean/customers.csv")
products = pd.read_csv(ROOT/"data/clean/products.csv")
orders = pd.read_csv(ROOT/"data/clean/orders.csv")
segments = pd.read_csv(ROOT/"outputs/customer_segments.csv")
triage = pd.read_csv(ROOT/"outputs/triage_results.csv")
assert customers.customer_id.is_unique and len(customers) == 1200
assert products.product_id.is_unique and len(products) == 40
assert orders.order_id.is_unique
assert orders.customer_id.isin(customers.customer_id).all()
assert orders.product_id.isin(products.product_id).all()
assert len(segments) == 1200 and len(triage) == 164
assert sum(1 for _ in open(ROOT/"outputs/agent_runs.jsonl", encoding="utf-8")) == 7
with sqlite3.connect(ROOT/"data/instamart.db") as con:
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
assert {"customers", "products", "orders"}.issubset(tables)
metrics = json.loads((ROOT/"outputs/part_d_summary.json").read_text())
assert metrics["agent_scenarios"] == 7
assert metrics["rag_eval_overlap_10_accuracy"] >= metrics["rag_eval_overlap_0_accuracy"]
print("PASS: required files, row counts, unique IDs, joins, SQLite tables, seven agent runs, and RAG evaluation.")
