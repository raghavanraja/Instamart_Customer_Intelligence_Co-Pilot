import json
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                             classification_report, silhouette_score)
from sklearn.utils.class_weight import compute_sample_weight
from common import CLEAN, OUTPUTS, STATE

def main():
    orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
    customers = pd.read_csv(CLEAN / "customers.csv")
    completed = orders[orders.status == "completed"].copy()
    # Fixed temporal split: features end on 2025-08-31; next 30 days form the outcome window.
    snapshot = pd.Timestamp("2025-08-31 23:59:59")
    outcome_end = snapshot + pd.Timedelta(days=30)
    hist = completed[completed.order_ts <= snapshot]
    future = completed[(completed.order_ts > snapshot) & (completed.order_ts <= outcome_end)]
    agg = hist.groupby("customer_id").agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        average_order_value=("order_amount", "mean"),
        total_quantity=("quantity", "sum"),
        mean_delivery=("delivery_minutes", "mean"),
    )
    agg["recency_days"] = (snapshot - agg["last_order"]).dt.total_seconds() / 86400
    future_ids = set(future.customer_id)
    agg["lapsed_30d"] = (~agg.index.isin(future_ids)).astype(int)
    agg = agg.reset_index().merge(customers[["customer_id","city","age","gender"]], on="customer_id", how="left")
    # Customers without historical completed orders are retained with zero activity and lapse label.
    all_cust = customers[["customer_id","city","age","gender"]].merge(agg, on=["customer_id","city","age","gender"], how="left")
    for col in ["frequency","monetary","average_order_value","total_quantity","mean_delivery","recency_days"]:
        all_cust[col] = all_cust[col].fillna(0)
    all_cust["lapsed_30d"] = (~all_cust["customer_id"].isin(future_ids)).astype(int)

    rfm_cols = ["recency_days","frequency","monetary"]
    rfm = all_cust[rfm_cols].copy()
    scaler_cluster = StandardScaler()
    X_scaled = scaler_cluster.fit_transform(np.log1p(rfm.clip(lower=0)))
    scores = []
    best_k, best_score = 2, -1
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
        scores.append({"k":k, "silhouette":float(score), "inertia":float(km.inertia_)})
        if score > best_score:
            best_k, best_score = k, score
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    all_cust["segment_id"] = km.fit_predict(X_scaled)
    segment_profile = all_cust.groupby("segment_id")[rfm_cols].mean()
    # Human-readable segment names based on spend/frequency, with recency as a tie-breaker.
    ranked = segment_profile.sort_values(["monetary","frequency"], ascending=False)
    names = {}
    labels = ["Premium", "Frequent", "Occasional", "At Risk", "Low Activity", "New/No History", "Other"]
    for i, idx in enumerate(ranked.index):
        names[int(idx)] = labels[i] if i < len(labels) else f"Segment {idx}"
    all_cust["segment"] = all_cust["segment_id"].map(names)
    all_cust.to_csv(OUTPUTS / "customer_segments.csv", index=False)
    pd.DataFrame(scores).to_csv(OUTPUTS / "kmeans_selection.csv", index=False)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    pd.DataFrame({"customer_id":all_cust.customer_id, "pc1":pcs[:,0], "pc2":pcs[:,1],
                  "segment_id":all_cust.segment_id}).to_csv(OUTPUTS / "pca_projection.csv", index=False)

    feature_cols = ["frequency","monetary","average_order_value","total_quantity",
                    "mean_delivery","recency_days","age"]
    X = all_cust[feature_cols].replace([np.inf,-np.inf], np.nan)
    y = all_cust["lapsed_30d"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    prep = Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    lr_pipe = Pipeline([("prep",prep),("model",LogisticRegression(max_iter=2000, class_weight="balanced"))])
    lr_grid = GridSearchCV(lr_pipe, {"model__C":[0.1,1,10]}, cv=3, scoring="f1", n_jobs=1)
    lr_grid.fit(X_train,y_train)
    rf_pipe = Pipeline([("imputer",SimpleImputer(strategy="median")),
                        ("model",RandomForestClassifier(random_state=42,class_weight="balanced_subsample"))])
    rf_grid = GridSearchCV(rf_pipe, {"model__n_estimators":[150,250],
                                     "model__max_depth":[None,8]}, cv=3, scoring="f1", n_jobs=1)
    rf_grid.fit(X_train,y_train)
    models = {"logistic_regression":lr_grid.best_estimator_,"random_forest":rf_grid.best_estimator_}
    metrics = {}
    for name, model in models.items():
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:,1]
        metrics[name] = {
            "best_params": (lr_grid.best_params_ if name=="logistic_regression" else rf_grid.best_params_),
            "f1":float(f1_score(y_test,pred,zero_division=0)),
            "precision":float(precision_score(y_test,pred,zero_division=0)),
            "recall":float(recall_score(y_test,pred,zero_division=0)),
            "roc_auc":float(roc_auc_score(y_test,prob)) if y_test.nunique()>1 else None
        }
    best_name = max(metrics, key=lambda n: metrics[n]["f1"])
    best_model = models[best_name]
    # Class-balance experiment: compare balanced model with unweighted logistic regression.
    unbalanced = Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),
                           ("model",LogisticRegression(max_iter=2000))])
    unbalanced.fit(X_train,y_train)
    ub_pred = unbalanced.predict(X_test)
    metrics["unweighted_logistic_experiment"] = {
        "f1":float(f1_score(y_test,ub_pred,zero_division=0)),
        "precision":float(precision_score(y_test,ub_pred,zero_division=0)),
        "recall":float(recall_score(y_test,ub_pred,zero_division=0))
    }
    joblib.dump({"model":best_model,"features":feature_cols,"model_name":best_name,
                 "snapshot_date":str(snapshot.date()),"outcome_window_days":30},
                OUTPUTS / "lapse_model.joblib")
    all_cust["lapse_probability"] = best_model.predict_proba(X)[:,1]
    all_cust.sort_values("lapse_probability",ascending=False).head(200).to_csv(
        OUTPUTS / "retention_list.csv",index=False)
    if best_name == "random_forest":
        imp = best_model.named_steps["model"].feature_importances_
    else:
        imp = np.abs(best_model.named_steps["model"].coef_[0])
    pd.DataFrame({"feature":feature_cols,"importance":imp}).sort_values(
        "importance",ascending=False).to_csv(OUTPUTS / "feature_importance.csv",index=False)
    (OUTPUTS / "part_b_metrics.json").write_text(json.dumps({
        "selected_k":best_k,"selected_k_silhouette":best_score,"selected_model":best_name,
        "metrics":metrics,"target_definition":"No completed order in the 30 days after 2025-08-31",
        "feature_cutoff":"2025-08-31; no future-window data used as features"
    },indent=2),encoding="utf-8")
    print("Part B complete:", best_name, "K=", best_k, "lapse prevalence=", round(float(y.mean()),3))

if __name__ == "__main__":
    main()
