"""05_analytics.py - Commercial analytics layer: pipeline, marketing, fleet lifecycle,
revenue-at-risk ranking. Outputs analysis-ready CSVs consumed by Excel and Streamlit."""
import os, json, numpy as np, pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
D = f"{ROOT}/data"
TODAY = pd.Timestamp("2026-06-30")
leads = pd.read_csv(f"{D}/leads_scored.csv")
acc = pd.read_csv(f"{D}/accounts_scored.csv")
acts = pd.read_csv(f"{D}/activities.csv")

STAGE_PROB = {"Prospect": .10, "Qualified": .25, "Proposal": .45, "Negotiation": .70,
              "Won": 1.00, "Lost": 0.00}
SRC_COST = {"LinkedIn": 50, "Trade Shows": 200, "Website": 20, "Referral": 0,
            "Email": 15, "Partners": 30, "Outbound": 40}
FUNNEL = ["Prospect", "Qualified", "Proposal", "Negotiation", "Won"]
LEVEL = {s: i for i, s in enumerate(FUNNEL)}
# Documented assumption: for a lost deal the furthest stage reached is inferred from the
# loss reason (a deal lost on price got as far as Negotiation; one that never decided did not).
LOST_STAGE = {"No Decision / Stalled": "Prospect", "Timing Deferred": "Qualified",
              "Spec Mismatch": "Proposal", "Price / Budget": "Negotiation",
              "Competitor Won": "Negotiation"}

leads["Furthest_Stage"] = np.where(leads.Sales_Stage == "Lost",
                                   leads.Lost_Reason.map(LOST_STAGE).fillna("Prospect"),
                                   leads.Sales_Stage)
leads["Furthest_Level"] = leads.Furthest_Stage.map(LEVEL)
leads["Stage_Win_Prob"] = leads.Sales_Stage.map(STAGE_PROB)
leads["Weighted_Value_AED"] = (leads.Deal_Value_AED * leads.Stage_Win_Prob).round(0)
OPEN = ["Prospect", "Qualified", "Proposal", "Negotiation"]

# ------------------------------------------------------------------ 1. FUNNEL
# A lead that reached a later stage necessarily passed the earlier ones.
rows = []
for i, st in enumerate(FUNNEL):
    m = leads.Furthest_Level >= i
    rows.append({"Stage": st, "Leads_Reached": int(m.sum()),
                 "Value_AED": float(leads.loc[m, "Deal_Value_AED"].sum())})
funnel = pd.DataFrame(rows)
tot = funnel.Leads_Reached.iloc[0]
funnel["Pct_of_Total"] = (funnel.Leads_Reached / tot * 100).round(1)
funnel["Stage_Conversion_Pct"] = (funnel.Leads_Reached / funnel.Leads_Reached.shift(1) * 100).round(1)
funnel["Drop_Off_Pct"] = (100 - funnel.Stage_Conversion_Pct).round(1)
funnel["Leads_Lost_At_Stage"] = (funnel.Leads_Reached.shift(1) - funnel.Leads_Reached).fillna(0).astype(int)
funnel["Value_Lost_AED"] = (funnel.Value_AED.shift(1) - funnel.Value_AED).fillna(0).round(0)

won = leads[leads.Sales_Stage == "Won"]
lost = leads[leads.Sales_Stage == "Lost"]
opn = leads[leads.Sales_Stage.isin(OPEN)]
kpi = {
    "Closed_Won_Revenue_AED": float(won.Deal_Value_AED.sum()),
    "Open_Pipeline_Value_AED": float(opn.Deal_Value_AED.sum()),
    "Weighted_Pipeline_AED": float(opn.Weighted_Value_AED.sum()),
    "Win_Rate_Pct": round(len(won) / max(len(won) + len(lost), 1) * 100, 1),
    "Lead_Conversion_Rate_Pct": round(leads.Converted_to_Sale.mean() * 100, 1),
    "Avg_Deal_Size_AED": float(won.Deal_Value_AED.mean()),
    "Avg_Sales_Cycle_Days": float(won.Sales_Cycle_Days.mean()),
    "Lost_Revenue_AED": float(lost.Deal_Value_AED.sum()),
    "Lost_Revenue_Late_Stage_AED": float(lost.loc[lost.Days_in_Pipeline > 90, "Deal_Value_AED"].sum()),
    "High_Priority_Leads": int(opn.Lead_Priority.isin(["Very High", "High"]).sum()),
    "Total_Leads": int(len(leads)), "Open_Deals": int(len(opn)),
    "Active_Customer_Revenue_AED": float(acc.Annual_Spend_AED.sum()),
    "Revenue_at_Risk_AED": float(acc.Revenue_at_Risk_AED.sum()),
    "High_Risk_Accounts": int(acc.Risk_Category.isin(["Critical", "High"]).sum()),
    "Total_Accounts": int(len(acc)),
    "Avg_Churn_Probability_Pct": round(acc.Churn_Probability.mean(), 1),
}

# by salesperson / industry / emirate
def seg(by):
    g = leads.groupby(by).apply(lambda d: pd.Series({
        "Leads": len(d),
        "Won_Deals": int((d.Sales_Stage == "Won").sum()),
        "Lost_Deals": int((d.Sales_Stage == "Lost").sum()),
        "Win_Rate_Pct": round((d.Sales_Stage == "Won").sum() /
                              max(((d.Sales_Stage == "Won") | (d.Sales_Stage == "Lost")).sum(), 1) * 100, 1),
        "Closed_Won_AED": float(d.loc[d.Sales_Stage == "Won", "Deal_Value_AED"].sum()),
        "Open_Pipeline_AED": float(d.loc[d.Sales_Stage.isin(OPEN), "Deal_Value_AED"].sum()),
        "Weighted_Pipeline_AED": float(d.loc[d.Sales_Stage.isin(OPEN), "Weighted_Value_AED"].sum()),
        "Avg_Deal_Size_AED": float(d.loc[d.Sales_Stage == "Won", "Deal_Value_AED"].mean()),
        "Avg_Sales_Cycle_Days": float(d.loc[d.Sales_Stage == "Won", "Sales_Cycle_Days"].mean()),
        "Lost_Revenue_AED": float(d.loc[d.Sales_Stage == "Lost", "Deal_Value_AED"].sum()),
    }), include_groups=False).reset_index()
    return g.round(1)


by_sp = seg("Salesperson").sort_values("Closed_Won_AED", ascending=False)
by_ind = seg("Industry").sort_values("Closed_Won_AED", ascending=False)
by_em = seg("Emirate").sort_values("Closed_Won_AED", ascending=False)
by_stage = (leads[leads.Sales_Stage.isin(OPEN)].groupby("Sales_Stage")
            .agg(Deals=("Lead_ID", "count"), Pipeline_AED=("Deal_Value_AED", "sum"),
                 Weighted_AED=("Weighted_Value_AED", "sum")).reindex(OPEN).reset_index())
by_stage["Stage_Win_Prob_Pct"] = [STAGE_PROB[s] * 100 for s in by_stage.Sales_Stage]

lost_reasons = (lost.groupby("Lost_Reason").agg(Deals=("Lead_ID", "count"),
                                                Revenue_Lost_AED=("Deal_Value_AED", "sum"))
                .sort_values("Revenue_Lost_AED", ascending=False).reset_index())
lost_reasons["Pct_of_Lost_Revenue"] = (lost_reasons.Revenue_Lost_AED /
                                       lost_reasons.Revenue_Lost_AED.sum() * 100).round(1)

# --------------------------------------------------------------- 2. MARKETING
mk = leads.groupby("Lead_Source").apply(lambda d: pd.Series({
    "Lead_Volume": len(d),
    "Qualified_Plus": int((d.Furthest_Level >= 1).sum()),
    "Won_Deals": int((d.Sales_Stage == "Won").sum()),
    "Conversion_Rate_Pct": round(d.Converted_to_Sale.mean() * 100, 1),
    "Revenue_AED": float(d.loc[d.Sales_Stage == "Won", "Deal_Value_AED"].sum()),
    "Avg_Deal_Value_AED": float(d.loc[d.Sales_Stage == "Won", "Deal_Value_AED"].mean()),
    "Open_Pipeline_AED": float(d.loc[d.Sales_Stage.isin(OPEN), "Deal_Value_AED"].sum()),
    "Avg_Lead_Score": round(d.Lead_Score.mean(), 1),
}), include_groups=False).reset_index()
mk["Lead_Quality_Pct"] = (mk.Qualified_Plus / mk.Lead_Volume * 100).round(1)
mk["Cost_per_Lead_USD"] = mk.Lead_Source.map(SRC_COST)
mk["Total_Spend_USD"] = mk.Cost_per_Lead_USD * mk.Lead_Volume
mk["Cost_per_Acquisition_USD"] = (mk.Total_Spend_USD / mk.Won_Deals.replace(0, np.nan)).round(0)
GROSS_MARGIN = 0.12   # assumed blended gross margin on fleet deals (documented assumption)
mk["Gross_Margin_USD"] = (mk.Revenue_AED / 3.6725 * GROSS_MARGIN).round(0)
mk["ROI_Pct"] = np.where(mk.Total_Spend_USD > 0,
                         ((mk.Gross_Margin_USD - mk.Total_Spend_USD) / mk.Total_Spend_USD * 100).round(0),
                         np.nan)   # Referral has zero media cost -> ROI undefined, shown as n/a
mk["Margin_Return_per_USD"] = (mk.Gross_Margin_USD / mk.Total_Spend_USD.replace(0, float("nan"))).round(1)
mk["Revenue_Share_Pct"] = (mk.Revenue_AED / mk.Revenue_AED.sum() * 100).round(1)
mk = mk.sort_values("Revenue_AED", ascending=False).round(1)

# --------------------------------------------------- 3. REVENUE AT RISK RANK
rar = (acc[acc.Risk_Category.isin(["Critical", "High", "Medium"])]
       .sort_values("Revenue_at_Risk_AED", ascending=False)
       [["Account_ID", "Company_Name", "Industry", "Emirate", "Account_Value_Tier", "Fleet_Size",
         "Annual_Spend_AED", "Churn_Probability", "Revenue_at_Risk_AED", "Risk_Category",
         "Avg_Satisfaction_Score", "Service_Tickets_12M", "Competitor_Present", "Days_To_Expiry",
         "Account_Manager", "Churn_Drivers", "Retention_Action"]].reset_index(drop=True))
rar.insert(0, "Rank", range(1, len(rar) + 1))
rar["Cumulative_RaR_AED"] = rar.Revenue_at_Risk_AED.cumsum()
rar["Cumulative_Pct_of_RaR"] = (rar.Cumulative_RaR_AED / acc.Revenue_at_Risk_AED.sum() * 100).round(1)

# ----------------------------------------------- 4. FLEET LIFECYCLE / UPGRADE
a = acc.copy()
a["Renewal_Flag"] = (a.Days_To_Expiry.between(-30, 90))
a["Upgrade_Flag"] = (a.Fleet_Size > 10) & (a.Days_Since_Last_Purchase >= 1095)
a["CrossSell_Flag"] = a.Spend_Growth_Pct >= 50
a["Upgrade_Potential_AED"] = (a.Annual_Spend_AED * np.where(a.Upgrade_Flag, 0.35, 0.15) *
                              (1 + a.Fleet_Size / 200)).round(0)
a["Urgency_Score"] = (
    np.clip(100 - a.Days_To_Expiry.clip(lower=0) / 3.65, 0, 100) * 0.40
    + np.clip(a.Annual_Spend_AED / a.Annual_Spend_AED.max() * 100, 0, 100) * 0.35
    + np.clip(a.Churn_Probability, 0, 100) * 0.25).round(1)


def opp_type(r):
    t = []
    if r.Renewal_Flag: t.append("Renewal")
    if r.Upgrade_Flag: t.append("Fleet Upgrade")
    if r.CrossSell_Flag: t.append("Cross-Sell")
    return " + ".join(t)


def opp_action(r):
    if r.Renewal_Flag and r.Risk_Category in ("Critical", "High"):
        return (f"URGENT: renewal in {int(r.Days_To_Expiry)}d on an at-risk account - "
                "director-led renewal + service recovery")
    if r.Renewal_Flag:
        return f"Open renewal negotiation now ({int(r.Days_To_Expiry)}d to expiry); offer multi-year lock-in"
    if r.Upgrade_Flag:
        return (f"Pitch fleet refresh - {int(r.Fleet_Size)} units, last purchase "
                f"{int(r.Days_Since_Last_Purchase / 365)}y ago; lead with TCO / residual-value case")
    if r.CrossSell_Flag:
        return (f"Expanding account (+{r.Spend_Growth_Pct:.0f}% YoY) - attach telematics, "
                "service contract and finance package")
    return ""


opp = a[a.Renewal_Flag | a.Upgrade_Flag | a.CrossSell_Flag].copy()
opp["Opportunity_Type"] = opp.apply(opp_type, axis=1)
opp["Recommended_Action"] = opp.apply(opp_action, axis=1)
opp = (opp.sort_values("Urgency_Score", ascending=False)
       [["Account_ID", "Company_Name", "Industry", "Emirate", "Account_Value_Tier", "Fleet_Size",
         "Primary_Product", "Annual_Spend_AED", "Spend_Growth_Pct", "Contract_Expiry",
         "Days_To_Expiry", "Last_Purchase_Date", "Days_Since_Last_Purchase", "Churn_Probability",
         "Risk_Category", "Opportunity_Type", "Upgrade_Potential_AED", "Urgency_Score",
         "Account_Manager", "Recommended_Action"]].reset_index(drop=True))

kpi["Renewal_Opportunities"] = int(a.Renewal_Flag.sum())
kpi["Upgrade_Opportunities"] = int(a.Upgrade_Flag.sum())
kpi["CrossSell_Opportunities"] = int(a.CrossSell_Flag.sum())
kpi["Opportunity_Pipeline_AED"] = float(opp.Upgrade_Potential_AED.sum())
kpi["Renewal_Value_AED"] = float(a.loc[a.Renewal_Flag, "Annual_Spend_AED"].sum())

# --------------------------------------------------------- 5. INDUSTRY / RISK
ind_rev = (acc.groupby("Industry").agg(Accounts=("Account_ID", "count"),
                                       Annual_Revenue_AED=("Annual_Spend_AED", "sum"),
                                       Revenue_at_Risk_AED=("Revenue_at_Risk_AED", "sum"),
                                       Avg_Churn_Prob=("Churn_Probability", "mean"),
                                       Avg_Satisfaction=("Avg_Satisfaction_Score", "mean"))
           .reset_index().sort_values("Annual_Revenue_AED", ascending=False).round(1))
ind_rev["Risk_Ratio_Pct"] = (ind_rev.Revenue_at_Risk_AED / ind_rev.Annual_Revenue_AED * 100).round(1)
risk_dist = (acc.groupby("Risk_Category").agg(Accounts=("Account_ID", "count"),
                                              Annual_Spend_AED=("Annual_Spend_AED", "sum"),
                                              Revenue_at_Risk_AED=("Revenue_at_Risk_AED", "sum"))
             .reindex(["Critical", "High", "Medium", "Low"]).fillna(0).reset_index())

for n, d in [("funnel", funnel), ("pipeline_by_salesperson", by_sp), ("pipeline_by_industry", by_ind),
             ("pipeline_by_emirate", by_em), ("pipeline_by_stage", by_stage),
             ("lost_reasons", lost_reasons), ("marketing_performance", mk),
             ("revenue_at_risk", rar), ("fleet_opportunities", opp), ("industry_revenue", ind_rev),
             ("risk_distribution", risk_dist)]:
    d.to_csv(f"{D}/analytics_{n}.csv", index=False)
leads.to_csv(f"{D}/leads_scored.csv", index=False)
json.dump(kpi, open(f"{D}/kpi_summary.json", "w"), indent=2)

print(json.dumps({k: (round(v) if isinstance(v, float) else v) for k, v in kpi.items()}, indent=1))
print(f"Analytics complete. Opportunities: {len(opp)} | RaR accounts ranked: {len(rar)}")
