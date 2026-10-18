"""03_lead_scoring.py - Lead conversion propensity: Logistic Regression vs Random Forest.
Leakage guard: Sales_Stage / Conversion_Date / Sales_Cycle_Days / Lost_Reason are NEVER features.
"""
import os, json, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.makedirs(f"{ROOT}/models", exist_ok=True)
leads = pd.read_csv(f"{ROOT}/data/leads.csv")

NUM = ["Fleet_Size", "Units_Requested", "Deal_Value_AED", "Days_in_Pipeline",
       "Number_of_Sales_Calls", "Days_Since_Last_Contact"]
CAT = ["Industry", "Company_Size", "Lead_Source", "Product_Interest", "Budget_Allocated",
       "Emirate", "Salesperson_ID"]
BANNED = {"Sales_Stage", "Conversion_Date", "Sales_Cycle_Days", "Lost_Reason", "Converted_to_Sale"}
assert not (set(NUM + CAT) & BANNED), "Data leakage: post-outcome field in feature set"

# Train ONLY on closed leads: an open lead has no realised outcome, so labelling it 0 would
# teach the model that "still in progress" means "will not buy".
closed = leads[leads.Sales_Stage.isin(["Won", "Lost"])]
X, y = closed[NUM + CAT], closed["Converted_to_Sale"]
X_all = leads[NUM + CAT]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Training rows (closed leads): {len(closed):,} | open leads scored later: {len(leads)-len(closed):,}")

pre = ColumnTransformer([("n", StandardScaler(), NUM),
                         ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
models = {
    "Logistic Regression": Pipeline([("p", pre), ("m", LogisticRegression(max_iter=2000, C=1.0,
                                                                         class_weight="balanced"))]),
    "Random Forest": Pipeline([("p", pre), ("m", RandomForestClassifier(
        n_estimators=400, max_depth=9, min_samples_leaf=12, class_weight="balanced",
        random_state=42, n_jobs=-1))]),
}

perf, cms = {}, {}
for name, pipe in models.items():
    pipe.fit(Xtr, ytr)
    pr = pipe.predict(Xte); pp = pipe.predict_proba(Xte)[:, 1]
    cv = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                         scoring="roc_auc")
    perf[name] = {"Accuracy": accuracy_score(yte, pr), "Precision": precision_score(yte, pr),
                  "Recall": recall_score(yte, pr), "F1": f1_score(yte, pr),
                  "ROC_AUC": roc_auc_score(yte, pp),
                  "CV_ROC_AUC_mean": cv.mean(), "CV_ROC_AUC_std": cv.std(),
                  "Train_ROC_AUC": roc_auc_score(ytr, pipe.predict_proba(Xtr)[:, 1]),
                  "n_train": len(Xtr), "n_test": len(Xte)}
    tn, fp, fn, tp = confusion_matrix(yte, pr).ravel()
    cms[name] = {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    print(f"{name:22s} ROC-AUC {perf[name]['ROC_AUC']:.3f} | F1 {perf[name]['F1']:.3f} | "
          f"Acc {perf[name]['Accuracy']:.3f} | CV {cv.mean():.3f}+/-{cv.std():.3f}")

best = max(perf, key=lambda k: perf[k]["ROC_AUC"])
champion = models[best]
print(f"Champion model: {best}")

# feature importance
feat = list(NUM) + list(champion.named_steps["p"].named_transformers_["c"].get_feature_names_out(CAT))
rf = models["Random Forest"].named_steps["m"]
lr = models["Logistic Regression"].named_steps["m"]
imp = pd.DataFrame({"Feature": feat, "RF_Importance": rf.feature_importances_,
                    "LR_Coefficient": lr.coef_[0]}).sort_values("RF_Importance", ascending=False)
imp["RF_Importance"] = imp.RF_Importance.round(4); imp["LR_Coefficient"] = imp.LR_Coefficient.round(4)
imp.to_csv(f"{ROOT}/data/lead_feature_importance.csv", index=False)

# score the full book
scores = champion.predict_proba(X_all)[:, 1]
leads["Lead_Score"] = (scores * 100).round(1)
leads["Lead_Priority"] = pd.cut(leads.Lead_Score, [-1, 25, 45, 65, 101],
                                labels=["Low", "Medium", "High", "Very High"]).astype(str)
leads["Expected_Value_AED"] = (leads.Deal_Value_AED * scores).round(0)


def action(r):
    if r.Lead_Priority == "Very High":
        return "Call within 24h - route to senior AE, prepare fleet proposal"
    if r.Lead_Priority == "High":
        return ("Re-engage - contact stalled >30 days" if r.Days_Since_Last_Contact > 30
                else "Schedule demo / site visit this week")
    if r.Lead_Priority == "Medium":
        return "Nurture sequence - qualify budget & decision timeline"
    return "Automated nurture only - do not consume field-sales capacity"


leads["Recommended_Action"] = leads.apply(action, axis=1)
leads.to_csv(f"{ROOT}/data/leads_scored.csv", index=False)

open_stages = ["Prospect", "Qualified", "Proposal", "Negotiation"]
top50 = (leads[leads.Sales_Stage.isin(open_stages)]
         .sort_values(["Lead_Score", "Deal_Value_AED"], ascending=False).head(50)
         [["Lead_ID", "Company_Name", "Industry", "Emirate", "Lead_Source", "Product_Interest",
           "Deal_Value_AED", "Lead_Score", "Lead_Priority", "Expected_Value_AED", "Sales_Stage",
           "Last_Contact_Date", "Days_Since_Last_Contact", "Salesperson", "Recommended_Action"]]
         .reset_index(drop=True))
top50.insert(0, "Rank", range(1, len(top50) + 1))
top50.to_csv(f"{ROOT}/Top_50_Priority_Leads.csv", index=False)

joblib.dump(champion, f"{ROOT}/models/lead_scoring_model.pkl")
json.dump({"champion": best, "performance": perf, "confusion_matrix": cms,
           "features_numeric": NUM, "features_categorical": CAT,
           "excluded_leakage_fields": sorted(BANNED),
           "training_population": "closed leads only (Won/Lost)",
           "trained_on": str(pd.Timestamp.now().date())},
          open(f"{ROOT}/models/lead_model_metadata.json", "w"), indent=2)

print(f"Top-50 open pipeline value: AED {top50.Deal_Value_AED.sum():,.0f} | "
      f"expected AED {top50.Expected_Value_AED.sum():,.0f}")
print("Lead scoring complete.")
