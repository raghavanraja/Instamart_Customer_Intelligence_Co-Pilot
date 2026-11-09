# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds an offline customer analytics and support prototype using fictional Instamart-style data. It covers data cleaning and SQL analytics, RFM customer segmentation and lapse prediction, support-ticket triage, policy retrieval, and a guarded refund-agent simulation.

The project uses the supplied deterministic `generate_data.py` and offline `mock_llm.py`. Both starter files are preserved unchanged. No paid API or network connection is required.

## Reproduce

Requirements: Python 3.11 or newer; SQLite 3.25 or newer.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python run_all.py
```

`run_all.py` regenerates the fictional raw data, audits and cleans it, creates the SQLite database, runs the analytics and ML workflow, classifies tickets, evaluates policy retrieval, and runs the refund-agent scenarios.

Run the integrity checks after the pipeline:

```bash
python tests/validate_project.py
```

To publish your own public GitHub repository, create an empty repository on GitHub, then run these commands from this folder (replace the URL with your repository URL):

```bash
git init
git add .
git commit -m "Complete Instamart customer intelligence capstone"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

To run one stage independently, execute the relevant script from the repository root, for example `python src/part_b_model.py`.

## Part A

- Audits row counts, columns, missing values and duplicate rows in every raw CSV.
- Normalizes city/gender fields; validates age; converts numeric/date columns; removes duplicate or invalid order records; checks customer/product keys.
- Writes cleaned files to `data/clean/`.
- Creates `data/instamart.db` with `customers`, `products`, and `orders` tables and indexes.
- `sql/queries.sql` contains seven business queries. Pandas EDA outputs are saved in `outputs/part_a_*.csv` and `outputs/part_a_eda_report.txt`.

## Part B

- Builds RFM and related customer features using completed orders through 2025-08-31.
- Defines the lapse target as no completed order during the following 30-day outcome window. Future-window data is used only for the label, not as a model feature.
- Tests K-means values K=2 through K=8 using silhouette score, selects the best score, and exports a two-component PCA projection.
- Tunes Logistic Regression and Random Forest with cross-validation, reports precision/recall/F1/ROC-AUC, and compares balanced and unweighted Logistic Regression.
- Saves the selected model to `outputs/lapse_model.joblib`, segment assignments to `outputs/customer_segments.csv`, and the top 200 predicted-risk customers to `outputs/retention_list.csv`.

The lapse definition is a retrospective teaching label based on the supplied six-month synthetic data. It is not a validated production churn definition.

## Part C

- Masks email addresses and Indian mobile-number patterns before the mock classifier receives ticket text.
- Sends empty and very long tickets directly to human review.
- Uses a few-shot prompt, bounded retries for transient errors, JSON/category validation, and a token-budget demonstration.
- Routes fraud tickets to the security team, premium-segment tickets to priority support, and invalid outputs to human review.
- Writes `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv`, and `outputs/part_c_summary.json`.

## Part D

- Splits policy documents into 40-word chunks with 10-word overlap and retrieves using TF-IDF cosine similarity.
- Evaluates 14 supplied policy questions with overlap 0 and overlap 10, including two out-of-domain questions that should be refused.
- Implements five tools: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`.
- Runs seven bounded ReAct-style scenarios with six-step, time, and token limits; saves JSONL traces and a checkpoint file. The synthetic support request date is modeled as one day after the order timestamp so the seven-day policy rule can be exercised reproducibly.
- Compares a deterministic plan-execute baseline with the agent scenarios.

All refunds are simulated in local files only. The project does not contact customers, payment systems, or external services.

## Limitations and Next Steps

- `mock_llm.py` is a deterministic teaching stand-in, not a real language model. Its outputs and failure patterns are intentionally synthetic.
- The generated dataset is fictional and small; model scores must not be interpreted as real business performance.
- The lapse target uses a fixed temporal window and a limited observation period. A production system needs longer history, a business-approved churn definition, temporal validation, monitoring, and bias checks.
- The RAG implementation uses TF-IDF rather than a production embedding model or managed vector database. Retrieval and answer evaluation are limited to the provided policy questions.
- PII masking uses regular expressions and is not a complete privacy solution. Production deployment needs stronger detection, access controls, audit logs, and security review.
- Refund issuance is simulated. A real workflow requires authorization, idempotency, transaction reconciliation, human approval, and payment-provider integration.
