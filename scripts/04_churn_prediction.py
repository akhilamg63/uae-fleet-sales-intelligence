"""04_churn_prediction.py - Customer churn propensity + Revenue at Risk.
Leakage guard: Churn_Status is the target only; no post-churn field enters the feature set.
"""
import os, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
acc = pd.read_csv(f"{ROOT}/data/accounts.csv")
srv = pd.read_csv(f"{ROOT}/data/service.csv")

# observable service aggregates (recomputed from transactions, not copied from account row)
agg = srv.groupby("Account_ID").agg(
    Srv_Tickets=("Service_ID", "count"), Srv_Avg_Sat=("Satisfaction_Score", "mean"),
    Srv_Min_Sat=("Satisfaction_Score", "min"), Srv_Avg_Res=("Resolution_Time_Days", "mean"),
    Srv_Repeat=("Repeat_Issue", lambda s: (s == "Yes").mean()),
    Srv_Cost=("Cost_AED", "sum")).reset_index()
df = acc.merge(agg, on="Account_ID", how="left").fillna(
    {"Srv_Tickets": 0, "Srv_Avg_Sat": 5, "Srv_Min_Sat": 5, "Srv_Avg_Res": 0,
     "Srv_Repeat": 0, "Srv_Cost": 0})

NUM = ["Fleet_Size", "Annual_Spend_AED", "Spend_Growth_Pct", "Contract_Age_Years",
       "Days_To_Expiry", "Days_Since_Last_Purchase", "Days_Since_Last_Activity",
       "Srv_Tickets", "Srv_Avg_Sat", "Srv_Min_Sat", "Srv_Avg_Res", "Srv_Repeat", "Srv_Cost"]
CAT = ["Industry", "Company_Size", "Primary_Product", "Competitor_Present", "Emirate",
       "Account_Value_Tier"]
assert "Churn_Status" not in NUM + CAT

X, y = df[NUM + CAT], df["Churn_Status"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
pre = ColumnTransformer([("n", StandardScaler(), NUM),
                         ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
# Probabilities are CALIBRATED (isotonic, 5-fold): Revenue at Risk is a financial figure,
# so predicted probabilities must be true expectations, not rank scores.
base = {
    "Logistic Regression": Pipeline([("p", pre), ("m", LogisticRegression(max_iter=3000))]),
    "Random Forest": Pipeline([("p", pre), ("m", RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=8, random_state=42, n_jobs=-1))]),
}
models = {k: CalibratedClassifierCV(v, method="isotonic", cv=5) for k, v in base.items()}
THRESH = {}
perf, cms = {}, {}
for name, pipe in models.items():
    pipe.fit(Xtr, ytr)
    # operating threshold chosen on TRAIN only (max F1) -> no test-set leakage
    ptr = pipe.predict_proba(Xtr)[:, 1]
    grid = np.linspace(0.10, 0.70, 61)
    THRESH[name] = float(grid[np.argmax([f1_score(ytr, (ptr >= t).astype(int)) for t in grid])])
    pp = pipe.predict_proba(Xte)[:, 1]
    pr = (pp >= THRESH[name]).astype(int)
    cv = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring="roc_auc")
    perf[name] = {"Accuracy": accuracy_score(yte, pr), "Precision": precision_score(yte, pr),
                  "Recall": recall_score(yte, pr), "F1": f1_score(yte, pr),
                  "ROC_AUC": roc_auc_score(yte, pp), "CV_ROC_AUC_mean": cv.mean(),
                  "CV_ROC_AUC_std": cv.std(),
                  "Train_ROC_AUC": roc_auc_score(ytr, pipe.predict_proba(Xtr)[:, 1]),
                  "Operating_Threshold": THRESH[name],
                  "n_train": len(Xtr), "n_test": len(Xte)}
    tn, fp, fn, tp = confusion_matrix(yte, pr).ravel()
    cms[name] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    print(f"{name:22s} ROC-AUC {perf[name]['ROC_AUC']:.3f} | Recall {perf[name]['Recall']:.3f} | "
          f"F1 {perf[name]['F1']:.3f} | CV {cv.mean():.3f}+/-{cv.std():.3f}")

best = max(perf, key=lambda k: perf[k]["ROC_AUC"]); champ = models[best]
print("Champion model:", best)

fitted_lr = base["Logistic Regression"].fit(Xtr, ytr)
fitted_rf = base["Random Forest"].fit(Xtr, ytr)
feat = list(NUM) + list(fitted_rf.named_steps["p"].named_transformers_["c"].get_feature_names_out(CAT))
imp = pd.DataFrame({"Feature": feat,
                    "RF_Importance": fitted_rf.named_steps["m"].feature_importances_,
                    "LR_Coefficient": fitted_lr.named_steps["m"].coef_[0]}
                   ).sort_values("RF_Importance", ascending=False).round(4)
imp.to_csv(f"{ROOT}/data/churn_feature_importance.csv", index=False)

p = champ.predict_proba(X)[:, 1]
df["Churn_Probability"] = (p * 100).round(1)
df["Revenue_at_Risk_AED"] = (df.Annual_Spend_AED * p).round(0)


def risk(r):
    if r.Churn_Probability > 70 and r.Annual_Spend_AED > 200_000: return "Critical"
    if r.Churn_Probability > 50 and r.Annual_Spend_AED > 100_000: return "High"
    if r.Churn_Probability >= 30: return "Medium"
    return "Low"


df["Risk_Category"] = df.apply(risk, axis=1)


def drivers(r):
    d = []
    if r.Competitor_Present == "Yes": d.append("Competitor active in account")
    if r.Srv_Avg_Sat < 3.8: d.append(f"Low satisfaction ({r.Srv_Avg_Sat:.1f}/5)")
    if r.Srv_Tickets > 8: d.append(f"High service load ({int(r.Srv_Tickets)} tickets)")
    if r.Srv_Repeat > 0.25: d.append(f"Repeat failures ({r.Srv_Repeat:.0%})")
    if r.Days_Since_Last_Purchase > 365: d.append(f"No purchase in {int(r.Days_Since_Last_Purchase)}d")
    if r.Days_Since_Last_Activity > 90: d.append(f"No contact in {int(r.Days_Since_Last_Activity)}d")
    if r.Days_To_Expiry < 120: d.append(f"Contract expires in {int(r.Days_To_Expiry)}d")
    if r.Spend_Growth_Pct < -15: d.append(f"Spend down {abs(r.Spend_Growth_Pct):.0f}% YoY")
    return "; ".join(d[:4]) if d else "No material risk signal"


def retention(r):
    if r.Risk_Category == "Critical":
        return ("ESCALATE: commercial director call within 7 days; service-recovery plan + "
                "retention offer on renewal")
    if r.Risk_Category == "High":
        return ("Assign named account manager; QBR within 30 days; resolve open service issues "
                "before renewal talks")
    if r.Risk_Category == "Medium":
        return "Proactive check-in call + satisfaction survey; monitor at monthly review"
    return "Business as usual - keep in standard care programme"


df["Churn_Drivers"] = df.apply(drivers, axis=1)
df["Retention_Action"] = df.apply(retention, axis=1)
df.to_csv(f"{ROOT}/data/accounts_scored.csv", index=False)

joblib.dump(champ, f"{ROOT}/models/churn_model.pkl")
meta = json.load(open(f"{ROOT}/models/lead_model_metadata.json"))
json.dump({"lead_scoring": meta,
           "churn": {"champion": best, "performance": perf, "confusion_matrix": cms,
                     "features_numeric": NUM, "features_categorical": CAT,
                     "risk_rules": "Critical >70% & spend>200k; High >50% & spend>100k; "
                                   "Medium 30-50%; Low <30%",
                     "probability_calibration": "isotonic, 5-fold",
                     "trained_on": str(pd.Timestamp.now().date())}},
          open(f"{ROOT}/models/model_metadata.json", "w"), indent=2)

print(f"Total Revenue at Risk: AED {df.Revenue_at_Risk_AED.sum():,.0f}")
print(df.Risk_Category.value_counts().to_dict())
