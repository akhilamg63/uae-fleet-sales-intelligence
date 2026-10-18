"""
UAE B2B Automotive & Fleet Sales Intelligence Engine - Streamlit dashboard.
Run:  streamlit run app.py
ALL DATA IS SYNTHETIC (see README.md).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
DATA = ROOT / "data"
TODAY = pd.Timestamp("2026-06-30")
NAVY, TEAL, RED, AMBER, GREEN = "#1F3864", "#0E7C7B", "#C0392B", "#E67E22", "#27AE60"
RISK_COLORS = {"Critical": RED, "High": AMBER, "Medium": "#F1C40F", "Low": GREEN}
STAGE_PROB = {"Prospect": .10, "Qualified": .25, "Proposal": .45, "Negotiation": .70}
OPEN_STAGES = list(STAGE_PROB)
FUNNEL = ["Prospect", "Qualified", "Proposal", "Negotiation", "Won"]

st.set_page_config(page_title="UAE Fleet Sales Intelligence", page_icon="🚚", layout="wide")


@st.cache_data
def load():
    leads = pd.read_csv(DATA / "leads_scored.csv", parse_dates=["Lead_Created_Date", "Last_Contact_Date"])
    acc = pd.read_csv(DATA / "accounts_scored.csv", parse_dates=["Contract_Start", "Contract_Expiry",
                                                                 "Last_Purchase_Date"])
    acts = pd.read_csv(DATA / "activities.csv", parse_dates=["Activity_Date"])
    srv = pd.read_csv(DATA / "service.csv", parse_dates=["Service_Date"])
    opp = pd.read_csv(DATA / "analytics_fleet_opportunities.csv")
    mk = pd.read_csv(DATA / "analytics_marketing_performance.csv")
    lost = pd.read_csv(DATA / "analytics_lost_reasons.csv")
    meta = json.load(open(ROOT / "models" / "model_metadata.json"))
    leads["Weighted_Value_AED"] = leads.Deal_Value_AED * leads.Sales_Stage.map(STAGE_PROB).fillna(0)
    return leads, acc, acts, srv, opp, mk, lost, meta


LEADS, ACC, ACTS, SRV, OPP, MK, LOST, META = load()


def aed(x, dp=0):
    if pd.isna(x):
        return "n/a"
    for unit, div in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(x) >= div:
            return f"AED {x / div:,.{2 if unit == 'B' else 1}f}{unit}"
    return f"AED {x:,.{dp}f}"


def reached(stage, reason=None):
    lost_max = {"Price / Budget": "Negotiation", "Competitor Won": "Negotiation",
                "Spec Mismatch": "Proposal", "Timing Deferred": "Qualified",
                "No Decision / Stalled": "Prospect"}
    if stage == "Won":
        return FUNNEL
    if stage == "Lost":
        return FUNNEL[:FUNNEL.index(lost_max.get(reason, "Prospect")) + 1]
    return FUNNEL[:FUNNEL.index(stage) + 1]


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🚚 Fleet Sales Intelligence")
st.sidebar.caption("Synthetic data · as-of 30 Jun 2026")
PAGE = st.sidebar.radio("Page", ["Executive Overview", "Lead Intelligence", "Pipeline Analytics",
                                 "Customer Churn & Risk", "Fleet Opportunities", "Marketing Analytics"])
st.sidebar.markdown("### Filters")
dmin, dmax = LEADS.Lead_Created_Date.min().date(), LEADS.Lead_Created_Date.max().date()
dr = st.sidebar.date_input("Lead created between", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(dr, tuple) and len(dr) == 2:
    d_from, d_to = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])
else:
    d_from, d_to = pd.Timestamp(dmin), pd.Timestamp(dmax)

ms = lambda label, opts: st.sidebar.multiselect(label, sorted(opts))
f_ind = ms("Industry", set(LEADS.Industry) | set(ACC.Industry))
f_em = ms("Emirate", set(LEADS.Emirate) | set(ACC.Emirate))
f_city = ms("City", set(LEADS.City) | set(ACC.City))
f_sp = ms("Salesperson / Account manager", set(LEADS.Salesperson) | set(ACC.Account_Manager))
f_src = ms("Lead source", LEADS.Lead_Source.unique())
f_prod = ms("Product / fleet type", set(LEADS.Product_Interest) | set(ACC.Primary_Product))
f_risk = ms("Risk level", ACC.Risk_Category.unique())
if st.sidebar.button("Reset filters"):
    st.rerun()


def filt_leads(df):
    m = df.Lead_Created_Date.between(d_from, d_to)
    for col, sel in [("Industry", f_ind), ("Emirate", f_em), ("City", f_city),
                     ("Salesperson", f_sp), ("Lead_Source", f_src), ("Product_Interest", f_prod)]:
        if sel:
            m &= df[col].isin(sel)
    return df[m]


def filt_acc(df):
    m = pd.Series(True, index=df.index)
    for col, sel in [("Industry", f_ind), ("Emirate", f_em), ("City", f_city),
                     ("Account_Manager", f_sp), ("Primary_Product", f_prod),
                     ("Risk_Category", f_risk)]:
        if sel:
            m &= df[col].isin(sel)
    return df[m]


L = filt_leads(LEADS)
A = filt_acc(ACC)
O = OPP[OPP.Account_ID.isin(A.Account_ID)]
if L.empty or A.empty:
    st.warning("No records match the current filters. Widen the selection in the sidebar.")
    st.stop()

won, lost_d = L[L.Sales_Stage == "Won"], L[L.Sales_Stage == "Lost"]
opn = L[L.Sales_Stage.isin(OPEN_STAGES)]


def kpi_row(items):
    for col, (label, value, help_) in zip(st.columns(len(items)), items):
        col.metric(label, value, help=help_)


def funnel_fig(df):
    counts = [df.apply(lambda r: st_ in reached(r.Sales_Stage, r.Lost_Reason), axis=1).sum()
              for st_ in FUNNEL]
    conv = [None] + [f"{counts[i] / counts[i - 1] * 100:.1f}% pass" if counts[i - 1] else ""
                     for i in range(1, len(counts))]
    fig = go.Figure(go.Funnel(y=FUNNEL, x=counts, textinfo="value+percent initial",
                              marker={"color": [NAVY, "#2E5A9C", TEAL, "#3FA796", GREEN]}))
    fig.update_layout(height=380, margin=dict(t=30, b=10), title="Sales funnel - leads reaching each stage")
    return fig, counts, conv


# ============================================================ 1. EXECUTIVE
if PAGE == "Executive Overview":
    st.title("Executive Overview")
    st.caption("Synthetic UAE fleet dealer group · every figure responds to the sidebar filters")
    kpi_row([("Closed-won revenue", aed(won.Deal_Value_AED.sum()), "Deals already won"),
             ("Open pipeline", aed(opn.Deal_Value_AED.sum()), "Live opportunity value"),
             ("Weighted pipeline", aed(opn.Weighted_Value_AED.sum()), "Value x stage win probability"),
             ("Win rate", f"{len(won) / max(len(won) + len(lost_d), 1) * 100:.1f}%", "Won / (Won+Lost)"),
             ("Lead conversion", f"{L.Converted_to_Sale.mean() * 100:.1f}%", "All leads converting")])
    kpi_row([("High-priority open leads", f"{int(opn.Lead_Priority.isin(['Very High', 'High']).sum()):,}",
              "Score above 45, still open"),
             ("Recurring base revenue", aed(A.Annual_Spend_AED.sum()), "Annual spend of active accounts"),
             ("Revenue at risk", aed(A.Revenue_at_Risk_AED.sum()), "Spend x churn probability"),
             ("High-risk accounts", f"{int(A.Risk_Category.isin(['Critical', 'High']).sum()):,}",
              "Critical + High"),
             ("Upgrade opportunities", f"{int(O.Opportunity_Type.str.contains('Upgrade').sum()):,}",
              "Fleet refresh due")])
    st.divider()
    c1, c2 = st.columns([1, 1])
    fig, counts, _ = funnel_fig(L)
    c1.plotly_chart(fig, use_container_width=True)
    ind = (A.groupby("Industry")[["Annual_Spend_AED", "Revenue_at_Risk_AED"]].sum()
           .sort_values("Annual_Spend_AED", ascending=False).head(8).reset_index())
    c2.plotly_chart(px.bar(ind, x="Annual_Spend_AED", y="Industry", orientation="h",
                           title="Recurring revenue by industry (top 8)",
                           color_discrete_sequence=[NAVY]).update_layout(height=380, yaxis_title=""),
                    use_container_width=True)
    stg = (opn.groupby("Sales_Stage").agg(Pipeline=("Deal_Value_AED", "sum"),
                                          Weighted=("Weighted_Value_AED", "sum"))
           .reindex(OPEN_STAGES).fillna(0).reset_index())
    fig2 = go.Figure()
    fig2.add_bar(x=stg.Sales_Stage, y=stg.Pipeline, name="Gross pipeline", marker_color=NAVY)
    fig2.add_bar(x=stg.Sales_Stage, y=stg.Weighted, name="Weighted", marker_color=TEAL)
    fig2.update_layout(barmode="group", height=360, title="Pipeline by stage - gross vs weighted (AED)")
    st.plotly_chart(fig2, use_container_width=True)
    biggest = int(np.argmin([counts[i] / counts[i - 1] if counts[i - 1] else 1
                             for i in range(1, len(counts))])) + 1
    st.info(f"**So what?** The largest leak is **{FUNNEL[biggest - 1]} → {FUNNEL[biggest]}** "
            f"({(1 - counts[biggest] / max(counts[biggest - 1], 1)) * 100:.0f}% drop-off). "
            f"Weighted pipeline of {aed(opn.Weighted_Value_AED.sum())} against "
            f"{aed(A.Revenue_at_Risk_AED.sum())} of revenue at risk means retention currently "
            f"protects more value than new business creates - fund both, but retain first.")

# ======================================================== 2. LEAD INTELLIGENCE
elif PAGE == "Lead Intelligence":
    st.title("Lead Intelligence")
    lp = META["lead_scoring"]["performance"][META["lead_scoring"]["champion"]]
    st.caption(f"Champion model: {META['lead_scoring']['champion']} · test ROC-AUC {lp['ROC_AUC']:.3f} "
               f"· trained on closed leads only · scores are conversion probabilities")
    kpi_row([("Open leads", f"{len(opn):,}", None),
             ("Very high priority", f"{int((opn.Lead_Priority == 'Very High').sum()):,}", None),
             ("Expected value of open book", aed(opn.Expected_Value_AED.sum()),
              "Sum of deal value x conversion probability"),
             ("Avg lead score (open)", f"{opn.Lead_Score.mean():.1f}", None)])
    st.subheader("Top 50 priority leads")
    top = opn.sort_values(["Lead_Score", "Deal_Value_AED"], ascending=False).head(50)
    st.dataframe(top[["Company_Name", "Industry", "Emirate", "Lead_Source", "Product_Interest",
                      "Deal_Value_AED", "Lead_Score", "Lead_Priority", "Expected_Value_AED",
                      "Sales_Stage", "Days_Since_Last_Contact", "Salesperson", "Recommended_Action"]],
                 use_container_width=True, hide_index=True,
                 column_config={"Deal_Value_AED": st.column_config.NumberColumn("Deal value", format="%,.0f"),
                                "Expected_Value_AED": st.column_config.NumberColumn("Expected value", format="%,.0f"),
                                "Lead_Score": st.column_config.ProgressColumn("Score", min_value=0,
                                                                             max_value=100, format="%.0f")})
    st.download_button("Download this list (CSV)", top.to_csv(index=False),
                       "top_priority_leads_filtered.csv", "text/csv")
    c1, c2 = st.columns(2)
    pr = opn.Lead_Priority.value_counts().reindex(["Very High", "High", "Medium", "Low"]).fillna(0).reset_index()
    pr.columns = ["Priority", "Leads"]
    c1.plotly_chart(px.bar(pr, x="Priority", y="Leads", title="Open leads by priority band",
                           color="Priority", color_discrete_sequence=[GREEN, TEAL, "#F1C40F", "#95A5A6"]
                           ).update_layout(height=340, showlegend=False), use_container_width=True)
    src = (L.groupby("Lead_Source").agg(Leads=("Lead_ID", "count"),
                                        Conversion=("Converted_to_Sale", "mean"),
                                        Avg_Score=("Lead_Score", "mean")).reset_index())
    src["Conversion"] = (src.Conversion * 100).round(1)
    c2.plotly_chart(px.scatter(src, x="Leads", y="Conversion", size="Avg_Score", color="Lead_Source",
                               title="Lead source: volume vs conversion (%)", text="Lead_Source"
                               ).update_traces(textposition="top center").update_layout(height=340,
                                                                                       showlegend=False),
                    use_container_width=True)
    st.subheader("All filtered leads")
    st.dataframe(L[["Lead_ID", "Company_Name", "Industry", "Lead_Source", "Deal_Value_AED",
                    "Lead_Score", "Lead_Priority", "Sales_Stage", "Last_Contact_Date",
                    "Salesperson"]].sort_values("Lead_Score", ascending=False),
                 use_container_width=True, hide_index=True, height=300)

# ======================================================= 3. PIPELINE ANALYTICS
elif PAGE == "Pipeline Analytics":
    st.title("Pipeline Analytics")
    cycle = won.Sales_Cycle_Days.mean()
    kpi_row([("Win rate", f"{len(won) / max(len(won) + len(lost_d), 1) * 100:.1f}%", None),
             ("Avg deal size (won)", aed(won.Deal_Value_AED.mean()), None),
             ("Avg sales cycle", f"{cycle:.0f} days" if not np.isnan(cycle) else "n/a", None),
             ("Lost revenue", aed(lost_d.Deal_Value_AED.sum()), "Value of deals marked Lost"),
             ("Weighted pipeline", aed(opn.Weighted_Value_AED.sum()), None)])
    fig, counts, _ = funnel_fig(L)
    st.plotly_chart(fig, use_container_width=True)
    drop = pd.DataFrame({"Transition": [f"{FUNNEL[i-1]} → {FUNNEL[i]}" for i in range(1, len(FUNNEL))],
                         "Pass_rate_%": [round(counts[i] / counts[i - 1] * 100, 1) if counts[i - 1] else 0
                                         for i in range(1, len(FUNNEL))]})
    drop["Drop_off_%"] = (100 - drop["Pass_rate_%"]).round(1)
    st.dataframe(drop, use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    sp = (L.groupby("Salesperson").apply(lambda d: pd.Series({
        "Won_AED": d.loc[d.Sales_Stage == "Won", "Deal_Value_AED"].sum(),
        "Deals_Won": (d.Sales_Stage == "Won").sum(),
        "Win_Rate": (d.Sales_Stage == "Won").sum() /
                    max(((d.Sales_Stage == "Won") | (d.Sales_Stage == "Lost")).sum(), 1) * 100,
        "Avg_Deal": d.loc[d.Sales_Stage == "Won", "Deal_Value_AED"].mean(),
        "Avg_Cycle": d.loc[d.Sales_Stage == "Won", "Sales_Cycle_Days"].mean()}),
        include_groups=False).reset_index().sort_values("Won_AED", ascending=False))
    c1.plotly_chart(px.bar(sp, x="Salesperson", y="Won_AED", color="Win_Rate",
                           title="Closed-won revenue by salesperson",
                           color_continuous_scale=["#C0392B", "#F1C40F", GREEN]
                           ).update_layout(height=380), use_container_width=True)
    ind = (L[L.Sales_Stage == "Won"].groupby("Industry")
           .agg(Revenue=("Deal_Value_AED", "sum"), Cycle=("Sales_Cycle_Days", "mean")).reset_index())
    c2.plotly_chart(px.scatter(ind, x="Cycle", y="Revenue", size="Revenue", color="Industry",
                               title="Revenue vs average sales cycle by industry"
                               ).update_layout(height=380, showlegend=False), use_container_width=True)
    st.dataframe(sp.round(1), use_container_width=True, hide_index=True)
    st.subheader("Why deals are lost")
    lr = (lost_d.groupby("Lost_Reason").agg(Deals=("Lead_ID", "count"),
                                            Revenue_Lost=("Deal_Value_AED", "sum"))
          .sort_values("Revenue_Lost", ascending=False).reset_index())
    c3, c4 = st.columns([2, 1])
    c3.plotly_chart(px.bar(lr, x="Lost_Reason", y="Revenue_Lost", color_discrete_sequence=[RED],
                           title="Revenue lost by reason (AED)").update_layout(height=340),
                    use_container_width=True)
    c4.dataframe(lr, use_container_width=True, hide_index=True)
    if len(lr):
        st.info(f"**So what?** {lr.iloc[0].Lost_Reason} accounts for {aed(lr.iloc[0].Revenue_Lost)} of "
                f"lost value across {int(lr.iloc[0].Deals)} deals. The steepest funnel leak is "
                f"{drop.sort_values('Drop_off_%', ascending=False).iloc[0].Transition} - fix that stage "
                "before adding more leads at the top.")

# ==================================================== 4. CHURN & RISK
elif PAGE == "Customer Churn & Risk":
    st.title("Customer Churn & Revenue at Risk")
    cp = META["churn"]["performance"][META["churn"]["champion"]]
    st.caption(f"Champion model: {META['churn']['champion']} · test ROC-AUC {cp['ROC_AUC']:.3f} · "
               "probabilities isotonic-calibrated so Revenue at Risk is a true expectation")
    kpi_row([("Accounts", f"{len(A):,}", None),
             ("Recurring revenue", aed(A.Annual_Spend_AED.sum()), None),
             ("Revenue at risk", aed(A.Revenue_at_Risk_AED.sum()), "Annual spend x churn probability"),
             ("Critical accounts", f"{int((A.Risk_Category == 'Critical').sum()):,}", None),
             ("Avg churn probability", f"{A.Churn_Probability.mean():.1f}%", None)])
    c1, c2 = st.columns(2)
    rd = A.Risk_Category.value_counts().reindex(["Critical", "High", "Medium", "Low"]).fillna(0).reset_index()
    rd.columns = ["Risk", "Accounts"]
    c1.plotly_chart(px.pie(rd, names="Risk", values="Accounts", hole=.45, title="Churn risk distribution",
                           color="Risk", color_discrete_map=RISK_COLORS).update_layout(height=360),
                    use_container_width=True)
    ir = (A.groupby("Industry").Revenue_at_Risk_AED.sum().sort_values(ascending=False)
          .head(8).reset_index())
    c2.plotly_chart(px.bar(ir, x="Revenue_at_Risk_AED", y="Industry", orientation="h",
                           title="Revenue at risk by industry (top 8)", color_discrete_sequence=[RED]
                           ).update_layout(height=360, yaxis_title=""), use_container_width=True)
    st.plotly_chart(px.scatter(A, x="Avg_Satisfaction_Score", y="Churn_Probability",
                               size="Annual_Spend_AED", color="Industry",
                               hover_name="Company_Name",
                               title="Service health vs churn probability (bubble = annual spend)"
                               ).update_layout(height=420), use_container_width=True)
    st.subheader("High-risk accounts ranked by exposure")
    hr = (A[A.Risk_Category.isin(["Critical", "High"])]
          .sort_values("Revenue_at_Risk_AED", ascending=False))
    st.dataframe(hr[["Company_Name", "Industry", "Account_Value_Tier", "Annual_Spend_AED",
                     "Churn_Probability", "Revenue_at_Risk_AED", "Risk_Category", "Churn_Drivers",
                     "Retention_Action", "Account_Manager"]], use_container_width=True, hide_index=True,
                 height=380,
                 column_config={"Annual_Spend_AED": st.column_config.NumberColumn("Annual spend", format="%,.0f"),
                                "Revenue_at_Risk_AED": st.column_config.NumberColumn("Revenue at risk", format="%,.0f"),
                                "Churn_Probability": st.column_config.ProgressColumn("Churn %", min_value=0,
                                                                                     max_value=100, format="%.0f")})
    top20 = hr.head(20).Revenue_at_Risk_AED.sum()
    if A.Revenue_at_Risk_AED.sum() > 0:
        st.info(f"**So what?** The top 20 exposed accounts carry {aed(top20)} - "
                f"{top20 / A.Revenue_at_Risk_AED.sum() * 100:.0f}% of all revenue at risk. "
                "Retention effort is worth concentrating there rather than spreading it evenly.")

# ==================================================== 5. FLEET OPPORTUNITIES
elif PAGE == "Fleet Opportunities":
    st.title("Fleet Lifecycle Opportunities")
    st.caption("Renewal: contract expiring within 90 days · Upgrade: fleet >10 units with no purchase "
               "in 3+ years · Cross-sell: spend up 50%+ YoY")
    ren = O[O.Opportunity_Type.str.contains("Renewal")]
    upg = O[O.Opportunity_Type.str.contains("Upgrade")]
    xs = O[O.Opportunity_Type.str.contains("Cross-Sell")]
    kpi_row([("Total opportunities", f"{len(O):,}", None),
             ("Renewals due", f"{len(ren):,}", "Contracts expiring within 90 days"),
             ("Fleet upgrades", f"{len(upg):,}", None),
             ("Cross-sell", f"{len(xs):,}", None),
             ("Opportunity value", aed(O.Upgrade_Potential_AED.sum()), "Modelled incremental revenue")])
    tabs = st.tabs(["Renewals", "Upgrades", "Cross-sell", "All ranked by urgency"])
    cols = ["Company_Name", "Industry", "Account_Value_Tier", "Fleet_Size", "Annual_Spend_AED",
            "Contract_Expiry", "Days_To_Expiry", "Days_Since_Last_Purchase", "Spend_Growth_Pct",
            "Churn_Probability", "Risk_Category", "Upgrade_Potential_AED", "Urgency_Score",
            "Account_Manager", "Recommended_Action"]
    for tab, df, note in zip(tabs, [ren, upg, xs, O],
                             ["Start these conversations before the customer starts them.",
                              "Fleet age is the trigger - lead with total cost of ownership.",
                              "Growing accounts absorb attachments: telematics, service, finance.",
                              "Ranked by urgency: 40% expiry proximity, 35% value, 25% churn risk."]):
        with tab:
            st.caption(note)
            st.dataframe(df[cols].sort_values("Urgency_Score", ascending=False),
                         use_container_width=True, hide_index=True, height=420)
    c1, c2 = st.columns(2)
    byt = O.assign(T=O.Opportunity_Type).groupby("T").Upgrade_Potential_AED.sum().reset_index()
    c1.plotly_chart(px.bar(byt, x="T", y="Upgrade_Potential_AED", title="Opportunity value by type (AED)",
                           color_discrete_sequence=[TEAL]).update_layout(height=360, xaxis_title=""),
                    use_container_width=True)
    if len(ren):
        c2.plotly_chart(px.histogram(ren, x="Days_To_Expiry", nbins=18, color_discrete_sequence=[NAVY],
                                     title="Renewal window - days to contract expiry"
                                     ).update_layout(height=360), use_container_width=True)

# ==================================================== 6. MARKETING ANALYTICS
else:
    st.title("Marketing Analytics")
    st.caption("Cost per lead is a planning assumption (LinkedIn $50, Trade Show $200, Website $20, "
               "Referral $0, Email $15, Partner $30, Outbound $40). ROI is on a 12% assumed gross margin.")
    src = (L.groupby("Lead_Source").apply(lambda d: pd.Series({
        "Lead_Volume": len(d),
        "Quality_Pct": d.apply(lambda r: "Qualified" in reached(r.Sales_Stage, r.Lost_Reason),
                               axis=1).mean() * 100,
        "Conversion_Pct": d.Converted_to_Sale.mean() * 100,
        "Revenue_AED": d.loc[d.Sales_Stage == "Won", "Deal_Value_AED"].sum(),
        "Avg_Deal_AED": d.loc[d.Sales_Stage == "Won", "Deal_Value_AED"].mean()}),
        include_groups=False).reset_index())
    cost = MK.set_index("Lead_Source").Cost_per_Lead_USD.to_dict()
    src["Cost_per_Lead_USD"] = src.Lead_Source.map(cost)
    src["Spend_USD"] = src.Cost_per_Lead_USD * src.Lead_Volume
    src["Gross_Margin_USD"] = src.Revenue_AED / 3.6725 * 0.12
    src["ROI_Pct"] = np.where(src.Spend_USD > 0,
                              (src.Gross_Margin_USD - src.Spend_USD) / src.Spend_USD * 100, np.nan)
    src = src.sort_values("Revenue_AED", ascending=False).round(1)
    kpi_row([("Sources", f"{len(src)}", None),
             ("Total leads", f"{int(src.Lead_Volume.sum()):,}", None),
             ("Attributed revenue", aed(src.Revenue_AED.sum()), None),
             ("Media spend", f"${src.Spend_USD.sum():,.0f}", "Assumption-based"),
             ("Blended conversion", f"{L.Converted_to_Sale.mean() * 100:.1f}%", None)])
    st.dataframe(src, use_container_width=True, hide_index=True,
                 column_config={"Revenue_AED": st.column_config.NumberColumn("Revenue (AED)", format="%,.0f"),
                                "Avg_Deal_AED": st.column_config.NumberColumn("Avg deal (AED)", format="%,.0f"),
                                "ROI_Pct": st.column_config.NumberColumn("ROI %", format="%.0f")})
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.scatter(src, x="Cost_per_Lead_USD", y="Conversion_Pct", size="Revenue_AED",
                               color="Lead_Source", text="Lead_Source",
                               title="Efficiency: cost per lead vs conversion (bubble = revenue)"
                               ).update_traces(textposition="top center"
                                               ).update_layout(height=400, showlegend=False),
                    use_container_width=True)
    c2.plotly_chart(px.pie(src, names="Lead_Source", values="Revenue_AED", hole=.45,
                           title="Revenue attribution by source").update_layout(height=400),
                    use_container_width=True)
    st.plotly_chart(px.bar(src.sort_values("Quality_Pct", ascending=False), x="Lead_Source",
                           y=["Quality_Pct", "Conversion_Pct"], barmode="group",
                           title="Lead quality vs conversion by source (%)",
                           color_discrete_sequence=[NAVY, TEAL]).update_layout(height=380),
                    use_container_width=True)
    best = src.sort_values("Conversion_Pct", ascending=False).iloc[0]
    worst = src.sort_values("Conversion_Pct").iloc[0]
    st.info(f"**So what?** {best.Lead_Source} converts at {best.Conversion_Pct:.1f}% versus "
            f"{worst.Lead_Source} at {worst.Conversion_Pct:.1f}%. Shifting spend from the tail toward "
            f"{best.Lead_Source} raises conversion without increasing lead volume - fewer leads, more deals.")

# ======================================================= ACCOUNT DRILL-DOWN
st.divider()
st.subheader("🔎 Account drill-down")
names = ["- select an account -"] + sorted(A.Company_Name.tolist())
pick = st.selectbox("Any account in the filtered set", names, key="drill")
if pick != names[0]:
    r = A[A.Company_Name == pick].iloc[0]
    a_srv = SRV[SRV.Account_ID == r.Account_ID]
    a_act = ACTS[ACTS.Account_ID == r.Account_ID]
    o = OPP[OPP.Account_ID == r.Account_ID]
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Profile**\n\n{r.Company_Name}  \n{r.Industry}  \n{r.Company_Size}  \n"
                f"{r.City}, {r.Emirate}  \nContact: {r.Primary_Contact}  \nAM: {r.Account_Manager}")
    c2.markdown(f"**Fleet & spend**\n\nFleet: {int(r.Fleet_Size)} units  \nProduct: {r.Primary_Product}  \n"
                f"Annual spend: {aed(r.Annual_Spend_AED)}  \nPrior year: {aed(r.Prev_Year_Spend_AED)}  \n"
                f"Growth: {r.Spend_Growth_Pct:+.1f}%  \nTier: {r.Account_Value_Tier}")
    c3.markdown(f"**Contract**\n\nStart: {r.Contract_Start:%d %b %Y}  \nExpiry: {r.Contract_Expiry:%d %b %Y}  \n"
                f"Days to expiry: {int(r.Days_To_Expiry)}  \nTerm: {int(r.Contract_Term_Years)} yrs  \n"
                f"Last purchase: {r.Last_Purchase_Date:%d %b %Y}  \n"
                f"Competitor present: {r.Competitor_Present}")
    c4.metric("Churn probability", f"{r.Churn_Probability:.1f}%")
    c4.metric("Revenue at risk", aed(r.Revenue_at_Risk_AED))
    c4.markdown(f"<span style='background:{RISK_COLORS[r.Risk_Category]};color:white;padding:3px 10px;"
                f"border-radius:4px'>{r.Risk_Category} risk</span>", unsafe_allow_html=True)
    d1, d2 = st.columns([1, 1])
    d1.markdown(f"**Service health** — {int(r.Service_Tickets_12M)} tickets · CSAT "
                f"{r.Avg_Satisfaction_Score:.2f}/5 · avg resolution {r.Avg_Resolution_Days:.1f} days · "
                f"repeat-issue rate {r.Repeat_Issue_Rate:.0%}")
    if len(a_srv):
        d1.dataframe(a_srv[["Service_Date", "Service_Type", "Issue_Category", "Satisfaction_Score",
                            "Resolution_Time_Days", "Repeat_Issue"]].sort_values("Service_Date",
                                                                                 ascending=False).head(8),
                     use_container_width=True, hide_index=True)
    d2.markdown(f"**Recent engagement** — last contact {int(r.Days_Since_Last_Activity)} days ago")
    if len(a_act):
        d2.dataframe(a_act[["Activity_Date", "Activity_Type", "Outcome", "Salesperson_ID"]]
                     .sort_values("Activity_Date", ascending=False).head(8),
                     use_container_width=True, hide_index=True)
    st.success(f"**Churn drivers:** {r.Churn_Drivers}")
    st.warning(f"**Recommended retention action:** {r.Retention_Action}")
    if len(o):
        st.info(f"**Lifecycle opportunity ({o.iloc[0].Opportunity_Type}, urgency "
                f"{o.iloc[0].Urgency_Score:.0f}/100):** {o.iloc[0].Recommended_Action} · "
                f"modelled upside {aed(o.iloc[0].Upgrade_Potential_AED)}")

st.sidebar.caption("⚠️ Synthetic data for portfolio demonstration only.")
