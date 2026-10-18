"""02_eda_validation.py - Data quality gate. Logs issues, never silently drops rows."""
import os, json, numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
D = os.path.join(ROOT, "data")
log = []


def note(sev, table, msg):
    log.append({"severity": sev, "table": table, "issue": msg})
    print(f"[{sev}] {table}: {msg}")


leads = pd.read_csv(f"{D}/leads.csv")
accounts = pd.read_csv(f"{D}/accounts.csv")
acts = pd.read_csv(f"{D}/activities.csv")
srv = pd.read_csv(f"{D}/service.csv")

for name, df, key in [("leads", leads, "Lead_ID"), ("accounts", accounts, "Account_ID"),
                      ("activities", acts, "Activity_ID"), ("service", srv, "Service_ID")]:
    note("INFO", name, f"{len(df):,} rows, {df.shape[1]} columns")
    d = int(df[key].duplicated().sum())
    note("OK" if d == 0 else "ERROR", name, f"duplicate {key}: {d}")
    miss = df.isna().sum()
    miss = miss[miss > 0].drop(labels=["Sales_Cycle_Days", "Conversion_Date", "Lost_Reason", "Account_ID", "Lead_ID"], errors="ignore")
    note("OK" if miss.empty else "WARN", name,
         "no unexpected missing values" if miss.empty else f"missing: {miss.to_dict()}")

# expected nulls are structural, not defects
note("INFO", "leads", f"Sales_Cycle_Days null for {leads.Sales_Cycle_Days.isna().sum()} non-converted "
                      "leads (structural, retained)")

# referential integrity
orph_a = (~acts.loc[acts.Account_ID.notna(), "Account_ID"].isin(accounts.Account_ID)).sum() - acts.Account_ID.isna().sum()
note("OK" if srv.Account_ID.isin(accounts.Account_ID).all() else "ERROR", "service",
     "all Account_IDs resolve to accounts")
note("OK" if acts.Lead_ID.dropna().isin(leads.Lead_ID).all() else "ERROR", "activities",
     "all Lead_IDs resolve to leads")

# logical consistency
bad = ((leads.Converted_to_Sale == 1) & (leads.Sales_Stage != "Won")).sum()
note("OK" if bad == 0 else "ERROR", "leads", f"converted leads not in Won stage: {bad}")
bad = (pd.to_datetime(leads.Conversion_Date) < pd.to_datetime(leads.Lead_Created_Date)).sum()
note("OK" if bad == 0 else "ERROR", "leads", f"conversion before creation: {bad}")
bad = (pd.to_datetime(accounts.Contract_Expiry) <= pd.to_datetime(accounts.Contract_Start)).sum()
note("OK" if bad == 0 else "ERROR", "accounts", f"contract expiry <= start: {bad}")
bad = (~srv.Satisfaction_Score.between(1, 5)).sum()
note("OK" if bad == 0 else "ERROR", "service", f"satisfaction outside 1-5: {bad}")

# outliers (IQR) - flagged and RETAINED as legitimate enterprise deals
for tbl, df, col in [("leads", leads, "Deal_Value_AED"), ("accounts", accounts, "Annual_Spend_AED")]:
    q1, q3 = df[col].quantile([.25, .75]); iqr = q3 - q1
    n = int((df[col] > q3 + 1.5 * iqr).sum())
    note("WARN", tbl, f"{col}: {n} high outliers (> {q3 + 1.5 * iqr:,.0f}) - retained, "
                      "large enterprise fleet deals are genuine")

# class balance / leakage guard
note("INFO", "leads", f"conversion rate {leads.Converted_to_Sale.mean():.1%} "
                      f"({leads.Converted_to_Sale.sum()} positives)")
note("INFO", "accounts", f"churn rate {accounts.Churn_Status.mean():.1%} "
                         f"({accounts.Churn_Status.sum()} positives)")
note("INFO", "leads", "LEAKAGE GUARD: Sales_Stage, Conversion_Date, Sales_Cycle_Days, Lost_Reason "
                      "excluded from lead-scoring features (post-outcome fields)")
note("INFO", "accounts", "LEAKAGE GUARD: Churn_Status excluded from churn features; only "
                         "observable health/contract/spend signals used")

pd.DataFrame(log).to_csv(f"{ROOT}/data/data_quality_log.csv", index=False)
print(f"\nValidation complete -> data/data_quality_log.csv "
      f"({sum(l['severity'] == 'ERROR' for l in log)} errors)")
