"""06_export_excel.py - Builds Automotive_Sales_Engine.xlsx (15 sheets, formatted, charted).
Executive KPI cards are live Excel formulas over the raw/analysis sheets, so the workbook recalculates."""
import os, json
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule
from openpyxl.chart import BarChart, PieChart, Reference
from openpyxl.worksheet.table import Table, TableStyleInfo

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
D, OUT = f"{ROOT}/data", f"{ROOT}/Automotive_Sales_Engine.xlsx"
NAVY, GOLD, LIGHT, WHITE = "1F3552", "C8A24A", "EAF0F6", "FFFFFF"
RED, ORANGE, YELLOW, GREEN = "F4CCCC", "FCE5CD", "FFF2CC", "D9EAD3"
FONT, AED0, PCT1 = "Arial", '#,##0;(#,##0);-', '0.0'
thin = Side(style="thin", color="BFBFBF"); BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)

R = lambda n: pd.read_csv(f"{D}/{n}.csv")
leads, accounts, acts, svc = R("leads"), R("accounts"), R("activities"), R("service")
ls, acs = R("leads_scored"), R("accounts_scored")
funnel, by_stage = R("analytics_funnel"), R("analytics_pipeline_by_stage")
by_sp, by_ind, by_em = R("analytics_pipeline_by_salesperson"), R("analytics_pipeline_by_industry"), R("analytics_pipeline_by_emirate")
lost_r, mkt = R("analytics_lost_reasons"), R("analytics_marketing_performance")
rar, fleet = R("analytics_revenue_at_risk"), R("analytics_fleet_opportunities")
riskdist, indrev, dq = R("analytics_risk_distribution"), R("analytics_industry_revenue"), R("data_quality_log")
lfi, cfi = R("lead_feature_importance"), R("churn_feature_importance")
top50 = pd.read_csv(f"{ROOT}/Top_50_Priority_Leads.csv")
kpi = json.load(open(f"{D}/kpi_summary.json"))
meta = json.load(open(f"{ROOT}/models/model_metadata.json"))
wb = Workbook(); wb.remove(wb.active)

def style_header(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = Font(name=FONT, bold=True, color=WHITE, size=10)
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER
    ws.row_dimensions[row].height = 30

def widths(ws, df, start=1, maxw=42):
    for i, col in enumerate(df.columns):
        q = df[col].astype(str).str.len().quantile(.92)
        w = max(len(str(col)) + 3, int(q if q == q else 8) + 3)
        ws.column_dimensions[get_column_letter(start + i)].width = min(w, maxw)

def put_df(ws, df, r0=1, c0=1, title=None, number_cols=(), pct_cols=(), table_name=None):
    if title:
        ws.cell(row=r0, column=c0, value=title).font = Font(name=FONT, bold=True, size=12, color=NAVY)
        r0 += 1
    hdr = r0
    for j, col in enumerate(df.columns):
        ws.cell(row=hdr, column=c0 + j, value=str(col))
    style_header(ws, hdr, c0 + len(df.columns) - 1)
    for i, row in enumerate(df.itertuples(index=False), start=1):
        for j, v in enumerate(row):
            v = None if (isinstance(v, float) and pd.isna(v)) else v
            cell = ws.cell(row=hdr + i, column=c0 + j, value=v)
            cell.font = Font(name=FONT, size=10); cell.border = BORDER
            cn = df.columns[j]
            if cn in number_cols: cell.number_format = AED0
            elif cn in pct_cols: cell.number_format = PCT1
    last = hdr + len(df)
    if table_name:
        ref = f"{get_column_letter(c0)}{hdr}:{get_column_letter(c0+len(df.columns)-1)}{last}"
        t = Table(displayName=table_name, ref=ref)
        t.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
        ws.add_table(t)
    return hdr, last

money_cols = lambda df: [c for c in df.columns if any(k in c for k in ("AED", "Value", "Spend", "Revenue", "Cost", "USD"))]
pct_cols = lambda df: [c for c in df.columns if any(k in c for k in ("Pct", "Rate", "Score", "Prob"))]

def sheet(name):
    ws = wb.create_sheet(name); ws.sheet_view.showGridLines = False; return ws

def banner(ws, text, sub="", span=12):
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=span)
    c = ws.cell(row=1, column=1, value=text)
    c.font = Font(name=FONT, bold=True, size=15, color=WHITE)
    c.fill = PatternFill("solid", fgColor=NAVY)
    c.alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws.row_dimensions[1].height = 28
    if sub:
        ws.cell(row=2, column=1, value=sub).font = Font(name=FONT, italic=True, size=9, color="555555")

# ---------------------------------------------------------------- 1. READ ME
ws = sheet("Read Me")
banner(ws, "UAE B2B AUTOMOTIVE & FLEET SALES INTELLIGENCE ENGINE",
       "Commercial decision-support system  |  ALL DATA IS SYNTHETIC  |  As-of 30 June 2026", span=8)
LM = meta["lead_scoring"]; CM = meta["churn"]
readme = [
    ("SECTION", "CONTENT"),
    ("Purpose", "Integrated decision-support system for a UAE fleet/automotive dealer: scores leads, predicts customer churn, quantifies revenue at risk, surfaces fleet renewal/upgrade opportunities, diagnoses the sales funnel and ranks marketing sources by return."),
    ("Data disclaimer", "100% SYNTHETIC data generated from a seeded latent-driver model. No real company, person, contract or transaction is represented. Values in AED unless labelled USD."),
    ("Scope", f"{kpi['Total_Leads']:,.0f} leads | {kpi['Total_Accounts']:,.0f} active accounts | {len(acts):,} sales activities | {len(svc):,} service records."),
    ("Lead scoring model", f"Logistic Regression vs Random Forest, trained on closed leads only (Won/Lost). Champion: {LM['champion']} (test ROC-AUC {LM['performance'][LM['champion']]['ROC_AUC']:.3f}). Output: conversion probability 0-100 plus a priority band."),
    ("Churn model", f"Logistic Regression vs Random Forest on contract, spend, engagement and service-health features. Champion: {CM['champion']} (test ROC-AUC {CM['performance'][CM['champion']]['ROC_AUC']:.3f}), threshold tuned for recall."),
    ("Leakage control", "Post-outcome fields (Sales_Stage, Conversion_Date, Sales_Cycle_Days, Lost_Reason, Churn_Status) are excluded from features. Stratified train/test split with 5-fold CV; train and test AUC are both reported so overfitting is visible."),
    ("Revenue at Risk", "Revenue_at_Risk = Annual_Spend x Churn_Probability. A deterministic exposure measure that ranks retention effort by money rather than by probability alone."),
    ("Risk bands", "Critical: churn >70% and spend >AED 200k | High: >50% and spend >AED 100k | Medium: 30-50% | Low: <30%."),
    ("Fleet opportunities", "Renewal = contract expiring within 90 days. Upgrade = fleet >10 units with no purchase in 3+ years. Cross-sell = 50%+ YoY spend growth. Ranked by an urgency score blending account value, expiry proximity and churn probability."),
    ("Weighted pipeline", "Open deal value x stage win probability (Prospect 10%, Qualified 25%, Proposal 45%, Negotiation 70%)."),
    ("Funnel assumption", "For a lost deal the furthest stage reached is inferred from the loss reason: price/competitor = Negotiation, spec mismatch = Proposal, timing deferred = Qualified, no decision = Prospect."),
    ("Marketing ROI", "Assumed cost per lead: LinkedIn $50, Trade Show $200, Website $20, Referral $0, Email $15, Partner $30, Outbound $40. ROI is computed on a 12% assumed gross margin, not on gross revenue. Referral has no media cost so its ROI is undefined (n/a); judge it on conversion and deal size."),
    ("How to use it", "Start on the Executive Dashboard. Lead Scoring and Top 50 Leads drive today's calling list; Churn Prediction and Revenue at Risk drive retention; Fleet Opportunities drives the renewal/upgrade campaign; Pipeline and Marketing Analytics drive resource allocation."),
    ("Rebuild", "python scripts/01_generate_data.py, then 02 -> 03 -> 04 -> 05 -> 06. Interactive dashboard: streamlit run app.py"),
]
ws.column_dimensions["A"].width = 24; ws.column_dimensions["B"].width = 112
r = 4
for a, b in readme:
    head = (r == 4)
    ws.cell(row=r, column=1, value=a).font = Font(name=FONT, bold=True, size=10, color=NAVY)
    c = ws.cell(row=r, column=2, value=b)
    c.font = Font(name=FONT, bold=head, size=10); c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[r].height = 15 if head else max(15, 12.5 * (len(b) // 108 + 1))
    for col in (1, 2): ws.cell(row=r, column=col).border = BORDER
    r += 1
put_df(ws, dq, r0=r + 2, c0=1, title="Data quality log (from 02_eda_validation.py)")

# ---------------------------------------------------------------- 2-5 RAW
for nm, df, tbl in [("Leads Raw", leads, "tblLeads"), ("Accounts Raw", accounts, "tblAccounts"),
                    ("Activities Raw", acts, "tblActs"), ("Service Raw", svc, "tblSvc")]:
    ws = sheet(nm)
    put_df(ws, df, number_cols=money_cols(df), pct_cols=pct_cols(df), table_name=tbl)
    widths(ws, df); ws.freeze_panes = "A2"

# ---------------------------------------------------------------- 6. DICTIONARY
DESCR = {"Lead_ID": "Unique CRM lead identifier", "Account_ID": "Unique customer account identifier",
 "Company_Name": "Synthetic company name", "Industry": "Customer vertical", "Company_Size": "Employee band",
 "Fleet_Size": "Vehicles operated", "Lead_Source": "Marketing channel that generated the lead",
 "Product_Interest": "Vehicle category requested", "Units_Requested": "Vehicles in the enquiry",
 "Budget_Allocated": "Budget signed off (Yes/No)", "Budget_Range": "Deal value band",
 "Deal_Value_AED": "Estimated contract value", "Sales_Stage": "Current CRM stage",
 "Lead_Created_Date": "Enquiry creation date", "Last_Contact_Date": "Most recent touchpoint",
 "Days_in_Pipeline": "Age of the lead in days", "Number_of_Sales_Calls": "Logged sales calls",
 "Days_Since_Last_Contact": "Engagement recency", "Emirate": "UAE emirate", "City": "Area / industrial zone",
 "Salesperson_ID": "Owning rep code", "Salesperson": "Owning rep", "Sales_Cycle_Days": "Days from creation to win",
 "Converted_to_Sale": "Lead-scoring target (1 = Won)", "Conversion_Date": "Win date", "Lost_Reason": "Loss category",
 "Lead_Score": "Model conversion probability 0-100", "Lead_Priority": "Very High / High / Medium / Low",
 "Expected_Value_AED": "Deal value x lead score", "Recommended_Action": "Next best action for the rep",
 "Annual_Spend_AED": "Current annual account revenue", "Prev_Year_Spend_AED": "Prior-year revenue",
 "Spend_Growth_Pct": "YoY spend growth", "Contract_Start": "Contract commencement",
 "Contract_Term_Years": "Contract length", "Contract_Expiry": "Contract end date",
 "Contract_Age_Years": "Years elapsed on contract", "Days_To_Expiry": "Days until expiry",
 "Last_Purchase_Date": "Most recent vehicle purchase", "Days_Since_Last_Purchase": "Purchase recency",
 "Days_Since_Last_Activity": "Commercial contact recency", "Service_Tickets_12M": "Tickets in last 12 months",
 "Avg_Satisfaction_Score": "Mean CSAT 1-5", "Repeat_Issue_Rate": "Share of tickets that are repeat failures",
 "Avg_Resolution_Days": "Mean days to close a ticket", "Competitor_Present": "Competitor active in the account",
 "Primary_Contact": "Named buyer", "Account_Manager": "Owning account manager",
 "Account_Value_Tier": "Bronze / Silver / Gold / Platinum", "Churn_Status": "Churn-model target (1 = churned)",
 "Churn_Probability": "Model churn probability 0-100", "Revenue_at_Risk_AED": "Annual spend x churn probability",
 "Risk_Category": "Critical / High / Medium / Low", "Churn_Drivers": "Top explanatory drivers for the account",
 "Retention_Action": "Recommended retention play", "Activity_ID": "Activity record key",
 "Activity_Type": "Call / Email / Meeting / Proposal / Demo / Site Visit", "Activity_Date": "Date of touchpoint",
 "Duration_Minutes": "Interaction length", "Outcome": "Result of the touchpoint", "Notes": "Free-text log",
 "Service_ID": "Service record key", "Service_Date": "Date of service event", "Service_Type": "Job category",
 "Satisfaction_Score": "CSAT for the job 1-5", "Issue_Category": "Fault category",
 "Resolution_Time_Days": "Days to resolve", "Repeat_Issue": "Recurrence flag", "Cost_AED": "Service cost",
 "Primary_Product": "Main vehicle category on contract", "Account_Manager_ID": "Owning AM code",
 "Srv_Tickets": "Service tickets, full history", "Srv_Avg_Sat": "Mean CSAT, full history",
 "Srv_Min_Sat": "Worst CSAT recorded", "Srv_Avg_Res": "Mean resolution days", "Srv_Repeat": "Repeat-issue rate",
 "Srv_Cost": "Total service cost"}
dd = pd.DataFrame([{"Table": t, "Column": c, "Data_Type": str(df[c].dtype),
                    "Business_Meaning": DESCR.get(c, "Derived analytical field")}
                   for t, df in [("Leads", leads), ("Accounts", accounts), ("Activities", acts), ("Service", svc)]
                   for c in df.columns])
ws = sheet("Data Dictionary"); put_df(ws, dd, table_name="tblDict"); widths(ws, dd)
ws.freeze_panes = "A2"; ws.column_dimensions["D"].width = 72

# ---------------------------------------------------------------- 7. LEAD SCORING
lead_cols = ["Lead_ID", "Company_Name", "Industry", "Emirate", "City", "Lead_Source", "Product_Interest",
             "Company_Size", "Fleet_Size", "Deal_Value_AED", "Sales_Stage", "Days_in_Pipeline",
             "Number_of_Sales_Calls", "Days_Since_Last_Contact", "Salesperson", "Lead_Score", "Lead_Priority",
             "Expected_Value_AED", "Recommended_Action"]
lsx = ls[lead_cols].sort_values("Lead_Score", ascending=False)
ws = sheet("Lead Scoring")
h, last = put_df(ws, lsx, number_cols=["Deal_Value_AED", "Expected_Value_AED"], table_name="tblLeadScore")
widths(ws, lsx); ws.freeze_panes = "A2"
sc = get_column_letter(lead_cols.index("Lead_Score") + 1); pc = get_column_letter(lead_cols.index("Lead_Priority") + 1)
ws.conditional_formatting.add(f"{sc}2:{sc}{last}", ColorScaleRule(
    start_type="num", start_value=0, start_color=RED, mid_type="num", mid_value=50, mid_color=YELLOW,
    end_type="num", end_value=100, end_color=GREEN))
for val, col in [("Very High", GREEN), ("High", "DCE9D5"), ("Medium", YELLOW), ("Low", RED)]:
    ws.conditional_formatting.add(f"{pc}2:{pc}{last}", CellIsRule(operator="equal", formula=[f'"{val}"'],
                                                                 fill=PatternFill("solid", fgColor=col)))
# ---------------------------------------------------------------- 8. TOP 50
ws = sheet("Top 50 Leads")
banner(ws, "TOP 50 PRIORITY LEADS - OPEN PIPELINE, RANKED BY MODEL SCORE x DEAL VALUE", span=12)
put_df(ws, top50, r0=3, number_cols=money_cols(top50), table_name="tblTop50")
widths(ws, top50); ws.freeze_panes = "A4"

# ---------------------------------------------------------------- 9. CHURN
ch_cols = ["Account_ID", "Company_Name", "Industry", "Emirate", "Account_Value_Tier", "Fleet_Size",
           "Annual_Spend_AED", "Contract_Expiry", "Days_To_Expiry", "Avg_Satisfaction_Score",
           "Service_Tickets_12M", "Competitor_Present", "Days_Since_Last_Purchase", "Churn_Probability",
           "Risk_Category", "Revenue_at_Risk_AED", "Churn_Drivers", "Retention_Action", "Account_Manager"]
chx = acs[ch_cols].sort_values("Churn_Probability", ascending=False)
ws = sheet("Churn Prediction")
h, last = put_df(ws, chx, number_cols=["Annual_Spend_AED", "Revenue_at_Risk_AED"], table_name="tblChurn")
widths(ws, chx); ws.freeze_panes = "A2"
# Revenue at Risk is written as a LIVE formula so the workbook recalculates if a probability
# or an annual spend figure is edited.
_sp = get_column_letter(ch_cols.index("Annual_Spend_AED") + 1)
_cp = get_column_letter(ch_cols.index("Churn_Probability") + 1)
_rr = get_column_letter(ch_cols.index("Revenue_at_Risk_AED") + 1)
for _r in range(h + 1, last + 1):
    _c = ws.cell(row=_r, column=ch_cols.index("Revenue_at_Risk_AED") + 1,
                 value=f"={_sp}{_r}*{_cp}{_r}/100")
    _c.number_format = AED0; _c.font = Font(name=FONT, size=10); _c.border = BORDER
rc = get_column_letter(ch_cols.index("Risk_Category") + 1); cp = get_column_letter(ch_cols.index("Churn_Probability") + 1)
for val, col in [("Critical", RED), ("High", ORANGE), ("Medium", YELLOW), ("Low", GREEN)]:
    ws.conditional_formatting.add(f"{rc}2:{rc}{last}", CellIsRule(operator="equal", formula=[f'"{val}"'],
                                                                 fill=PatternFill("solid", fgColor=col)))
ws.conditional_formatting.add(f"{cp}2:{cp}{last}", ColorScaleRule(
    start_type="num", start_value=0, start_color=GREEN, mid_type="num", mid_value=50, mid_color=YELLOW,
    end_type="num", end_value=100, end_color=RED))

# ---------------------------------------------------------------- 10. REVENUE AT RISK
ws = sheet("Revenue at Risk")
banner(ws, "REVENUE AT RISK = ANNUAL SPEND x CHURN PROBABILITY (RANKED BY EXPOSURE)", span=14)
h, last = put_df(ws, rar, r0=3, number_cols=money_cols(rar), table_name="tblRaR")
widths(ws, rar); ws.freeze_panes = "A4"
rcol = get_column_letter(list(rar.columns).index("Risk_Category") + 1)
for val, col in [("Critical", RED), ("High", ORANGE), ("Medium", YELLOW), ("Low", GREEN)]:
    ws.conditional_formatting.add(f"{rcol}4:{rcol}{last}", CellIsRule(operator="equal", formula=[f'"{val}"'],
                                                                     fill=PatternFill("solid", fgColor=col)))

# ---------------------------------------------------------------- 11. FLEET
ws = sheet("Fleet Opportunities")
banner(ws, "RENEWAL / UPGRADE / CROSS-SELL OPPORTUNITIES - RANKED BY URGENCY", span=14)
h, last = put_df(ws, fleet, r0=3, number_cols=money_cols(fleet), table_name="tblFleet")
widths(ws, fleet); ws.freeze_panes = "A4"
ucol = get_column_letter(list(fleet.columns).index("Urgency_Score") + 1)
ws.conditional_formatting.add(f"{ucol}4:{ucol}{last}",
                              ColorScaleRule(start_type="min", start_color=WHITE, end_type="max", end_color=GOLD))

# ---------------------------------------------------------------- 12. PIPELINE
ws = sheet("Pipeline Analytics"); banner(ws, "SALES PIPELINE ANALYTICS", span=12)
hf, lf = put_df(ws, funnel, r0=3, title="Funnel - leads reaching each stage",
                number_cols=money_cols(funnel), pct_cols=pct_cols(funnel))
hst, lst = put_df(ws, by_stage, r0=lf + 3, title="Open pipeline by stage", number_cols=money_cols(by_stage))
hsp, lsp = put_df(ws, by_sp, r0=lst + 3, title="Performance by salesperson", number_cols=money_cols(by_sp))
hin, lin = put_df(ws, by_ind, r0=lsp + 3, title="Performance by industry", number_cols=money_cols(by_ind))
hem, lem = put_df(ws, by_em, r0=lin + 3, title="Performance by emirate", number_cols=money_cols(by_em))
hlr, llr = put_df(ws, lost_r, r0=lem + 3, title="Lost-deal analysis", number_cols=money_cols(lost_r))
for i, c in enumerate(funnel.columns):
    ws.column_dimensions[get_column_letter(i + 1)].width = max(17, len(c) + 3)

def bar(ws, title, col, hrow, lrow, catcol, anchor, maxcol=None, typ="col"):
    ch = BarChart(); ch.type = typ; ch.style = 10; ch.title = title
    ch.add_data(Reference(ws, min_col=col, max_col=maxcol or col, min_row=hrow, max_row=lrow), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=catcol, min_row=hrow + 1, max_row=lrow))
    ch.height, ch.width = 8.5, 15.5; ws.add_chart(ch, anchor)

bar(ws, "Sales funnel - leads reaching each stage", 2, hf, lf, 1, "K3", typ="bar")
bar(ws, "Open vs weighted pipeline by stage", 3, hst, lst, 1, "K20", maxcol=4)
bar(ws, "Closed-won revenue by salesperson", 6, hsp, lsp, 1, "K37")
bar(ws, "Lost revenue by reason", 3, hlr, llr, 1, "K54")

# ---------------------------------------------------------------- 13. MARKETING
ws = sheet("Marketing Analytics")
banner(ws, "LEAD SOURCE PERFORMANCE & MARKETING EFFICIENCY", span=14)
hm, lm = put_df(ws, mkt, r0=3, number_cols=money_cols(mkt), pct_cols=pct_cols(mkt), table_name="tblMkt")
widths(ws, mkt)
ws.cell(row=lm + 2, column=1, value="ROI is computed on a 12% assumed gross margin. Referral carries no media cost, "
        "so its ROI is undefined (n/a) - judge it on conversion rate and average deal size. "
        "CAVEAT: the prescribed cost-per-lead figures are nominal digital-marketing rates while "
        "the deals are enterprise fleet orders, so absolute ROI percentages are implausibly large. "
        "Use them to RANK channels against each other, not as a budget forecast."
        ).font = Font(name=FONT, italic=True, size=9, color="555555")
cidx = {c: i + 1 for i, c in enumerate(mkt.columns)}
bar(ws, "Revenue by lead source (AED)", cidx["Revenue_AED"], hm, lm, 1, f"A{lm+5}")
bar(ws, "Conversion rate vs lead quality by source (%)", cidx["Conversion_Rate_Pct"], hm, lm, 1, f"J{lm+5}",
    maxcol=cidx["Conversion_Rate_Pct"])
ch = BarChart(); ch.type = "col"; ch.grouping = "clustered"; ch.title = "Lead quality % by source"
ch.add_data(Reference(ws, min_col=cidx["Lead_Quality_Pct"], min_row=hm, max_row=lm), titles_from_data=True)
ch.set_categories(Reference(ws, min_col=1, min_row=hm + 1, max_row=lm))
ch.height, ch.width = 8.5, 15.5; ws.add_chart(ch, f"S{lm+5}")

# ---------------------------------------------------------------- 14. MODEL PERFORMANCE
ws = sheet("Model Performance")
banner(ws, "MODEL PERFORMANCE - HELD-OUT TEST SET (ACTUAL SCIKIT-LEARN OUTPUT)", span=13)
perf = pd.DataFrame([{"Task": task, "Model": m, "Champion": "YES" if m == meta[k]["champion"] else "",
                      "Accuracy": round(p["Accuracy"], 3), "Precision": round(p["Precision"], 3),
                      "Recall": round(p["Recall"], 3), "F1": round(p["F1"], 3),
                      "ROC_AUC_Test": round(p["ROC_AUC"], 3), "ROC_AUC_Train": round(p["Train_ROC_AUC"], 3),
                      "CV_ROC_AUC_Mean": round(p["CV_ROC_AUC_mean"], 3), "CV_Std": round(p["CV_ROC_AUC_std"], 3),
                      "n_train": p["n_train"], "n_test": p["n_test"]}
                     for task, k in [("Lead Scoring", "lead_scoring"), ("Churn", "churn")]
                     for m, p in meta[k]["performance"].items()])
hp, lp = put_df(ws, perf, r0=3); widths(ws, perf)
cmdf = pd.DataFrame([{"Task": task, "Model": m, "True_Negative": c["TN"], "False_Positive": c["FP"],
                      "False_Negative": c["FN"], "True_Positive": c["TP"],
                      "Precision_Check": round(c["TP"] / max(c["TP"] + c["FP"], 1), 3),
                      "Recall_Check": round(c["TP"] / max(c["TP"] + c["FN"], 1), 3)}
                     for task, k in [("Lead Scoring", "lead_scoring"), ("Churn", "churn")]
                     for m, c in meta[k]["confusion_matrix"].items()])
hcm, lcm = put_df(ws, cmdf, r0=lp + 3, title="Confusion matrices (test set)")
hfi, lfi_l = put_df(ws, lfi.head(15), r0=lcm + 3, title="Lead scoring - top 15 feature drivers")
hci, lci = put_df(ws, cfi.head(15), r0=lcm + 3, c0=5, title="Churn - top 15 feature drivers")
ws.cell(row=lci + 3, column=1, value="Interpretation: train ROC-AUC close to test ROC-AUC indicates the model is not "
        "overfitting. For churn, recall is prioritised over precision - missing a leaver costs far more than an "
        "unnecessary retention call.").font = Font(name=FONT, italic=True, size=9, color="555555")
bar(ws, "Lead scoring - feature importance (RF)", 2, hfi, lfi_l, 1, f"J{lcm+3}", typ="bar")
bar(ws, "Churn - feature importance (RF)", 6, hci, lci, 5, f"J{lcm+22}", typ="bar")

# ---------------------------------------------------------------- 15. DASHBOARD
ws = sheet("Executive Dashboard")
banner(ws, "EXECUTIVE DASHBOARD - UAE FLEET SALES INTELLIGENCE (SYNTHETIC DATA)", span=12)
ws.cell(row=2, column=1, value="As-of 30 Jun 2026 | AED unless stated | KPI cards are live formulas over the raw and "
        "analysis sheets").font = Font(name=FONT, italic=True, size=9, color="555555")
lcol = {c: get_column_letter(i + 1) for i, c in enumerate(leads.columns)}
scol = {c: get_column_letter(i + 1) for i, c in enumerate(lsx.columns)}
ccol = {c: get_column_letter(i + 1) for i, c in enumerate(chx.columns)}
fcol = {c: get_column_letter(i + 1) for i, c in enumerate(fleet.columns)}
NL, NS, NA, NF = len(leads) + 1, len(lsx) + 1, len(chx) + 1, len(fleet) + 3
ST, SG = lcol["Sales_Stage"], lcol["Deal_Value_AED"]
KPIS = [
 ("CLOSED-WON REVENUE", f"=SUMIFS('Leads Raw'!{SG}2:{SG}{NL},'Leads Raw'!{ST}2:{ST}{NL},\"Won\")", AED0),
 ("RECURRING BASE REVENUE", f"=SUM('Churn Prediction'!{ccol['Annual_Spend_AED']}2:{ccol['Annual_Spend_AED']}{NA})", AED0),
 ("OPEN PIPELINE", f"=SUMIFS('Leads Raw'!{SG}2:{SG}{NL},'Leads Raw'!{ST}2:{ST}{NL},\"<>Won\",'Leads Raw'!{ST}2:{ST}{NL},\"<>Lost\")", AED0),
 ("WEIGHTED PIPELINE", f"=SUM('Pipeline Analytics'!D{hst+1}:D{lst})", AED0),
 ("WIN RATE %", f"=ROUND(COUNTIF('Leads Raw'!{ST}2:{ST}{NL},\"Won\")/(COUNTIF('Leads Raw'!{ST}2:{ST}{NL},\"Won\")+COUNTIF('Leads Raw'!{ST}2:{ST}{NL},\"Lost\"))*100,1)", '0.0"%"'),
 ("LEAD CONVERSION %", f"=ROUND(COUNTIF('Leads Raw'!{ST}2:{ST}{NL},\"Won\")/{len(leads)}*100,1)", '0.0"%"'),
 ("HIGH-PRIORITY OPEN LEADS", f"=COUNTIFS('Lead Scoring'!{scol['Lead_Priority']}2:{scol['Lead_Priority']}{NS},\"Very High\",'Lead Scoring'!{scol['Sales_Stage']}2:{scol['Sales_Stage']}{NS},\"<>Won\",'Lead Scoring'!{scol['Sales_Stage']}2:{scol['Sales_Stage']}{NS},\"<>Lost\")", '#,##0'),
 ("REVENUE AT RISK", f"=SUM('Churn Prediction'!{ccol['Revenue_at_Risk_AED']}2:{ccol['Revenue_at_Risk_AED']}{NA})", AED0),
 ("HIGH-RISK ACCOUNTS", f"=COUNTIF('Churn Prediction'!{ccol['Risk_Category']}2:{ccol['Risk_Category']}{NA},\"Critical\")+COUNTIF('Churn Prediction'!{ccol['Risk_Category']}2:{ccol['Risk_Category']}{NA},\"High\")", '#,##0'),
 ("RENEWAL OPPORTUNITIES", f"=COUNTIF('Fleet Opportunities'!{fcol['Opportunity_Type']}4:{fcol['Opportunity_Type']}{NF},\"*Renewal*\")", '#,##0'),
 ("UPGRADE OPPORTUNITIES", f"=COUNTIF('Fleet Opportunities'!{fcol['Opportunity_Type']}4:{fcol['Opportunity_Type']}{NF},\"*Upgrade*\")", '#,##0'),
 ("OPPORTUNITY VALUE", f"=SUM('Fleet Opportunities'!{fcol['Upgrade_Potential_AED']}4:{fcol['Upgrade_Potential_AED']}{NF})", AED0),
]
for i, (label, formula, fmt) in enumerate(KPIS):
    row, col = 4 + (i // 4) * 3, 1 + (i % 4) * 3
    ws.merge_cells(start_row=row, start_column=col, end_row=row, end_column=col + 2)
    c = ws.cell(row=row, column=col, value=label)
    c.font = Font(name=FONT, bold=True, size=9, color=WHITE); c.fill = PatternFill("solid", fgColor=NAVY)
    c.alignment = Alignment(horizontal="center", vertical="center")
    ws.merge_cells(start_row=row + 1, start_column=col, end_row=row + 1, end_column=col + 2)
    v = ws.cell(row=row + 1, column=col, value=formula)
    v.font = Font(name=FONT, bold=True, size=14, color=NAVY); v.fill = PatternFill("solid", fgColor=LIGHT)
    v.alignment = Alignment(horizontal="center", vertical="center"); v.number_format = fmt
    ws.row_dimensions[row + 1].height = 26
for i in range(1, 13):
    ws.column_dimensions[get_column_letter(i)].width = 13

r = 15
ws.cell(row=r, column=1, value="CHART DATA (linked to the analysis sheets)").font = Font(name=FONT, bold=True, color=NAVY)
fh, fl = put_df(ws, funnel[["Stage", "Leads_Reached"]], r0=r + 1, c0=1)
top_ind = indrev.nlargest(8, "Annual_Revenue_AED")[["Industry", "Annual_Revenue_AED", "Revenue_at_Risk_AED"]]
ih, il = put_df(ws, top_ind, r0=r + 1, c0=4, number_cols=list(top_ind.columns[1:]))
mh, ml = put_df(ws, mkt[["Lead_Source", "Revenue_AED", "Lead_Volume"]], r0=r + 1, c0=8, number_cols=["Revenue_AED"])
sh2, sl2 = put_df(ws, by_stage[["Sales_Stage", "Pipeline_AED", "Weighted_AED"]], r0=r + 1, c0=12,
                  number_cols=["Pipeline_AED", "Weighted_AED"])
rh, rl = put_df(ws, rar.nlargest(10, "Revenue_at_Risk_AED")[["Company_Name", "Revenue_at_Risk_AED"]],
                r0=r + 1, c0=16, number_cols=["Revenue_at_Risk_AED"])
dh, dl = put_df(ws, riskdist[["Risk_Category", "Accounts"]], r0=r + 1, c0=19)
for c in range(13, 22):
    ws.column_dimensions[get_column_letter(c)].width = 18
r_ch = r + 14
bar(ws, "Sales funnel - leads reaching each stage", 2, fh, fl, 1, f"A{r_ch}", typ="bar")
bar(ws, "Revenue and revenue at risk by industry (top 8)", 5, ih, il, 4, f"F{r_ch}", maxcol=6)
bar(ws, "Lead source performance - revenue", 9, mh, ml, 8, f"L{r_ch}")
bar(ws, "Pipeline vs weighted pipeline by stage", 13, sh2, sl2, 12, f"A{r_ch+18}", maxcol=14)
bar(ws, "Top 10 accounts by revenue at risk", 17, rh, rl, 16, f"F{r_ch+18}", typ="bar")
pie = PieChart(); pie.title = "Accounts by churn risk category"
pie.add_data(Reference(ws, min_col=20, min_row=dh, max_row=dl), titles_from_data=True)
pie.set_categories(Reference(ws, min_col=19, min_row=dh + 1, max_row=dl))
pie.height, pie.width = 8.5, 15.5; ws.add_chart(pie, f"L{r_ch+18}")

for w in wb.worksheets:
    w.sheet_properties.tabColor = NAVY
wb.save(OUT)
print("Workbook written:", OUT, "| sheets:", len(wb.sheetnames))
