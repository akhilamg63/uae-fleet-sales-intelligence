# UAE B2B Automotive & Fleet Sales Intelligence Engine

▶ **[Open the live dashboard](https://akhilamg63.github.io/uae-fleet-sales-intelligence/)** — 18 charts, six tabs, no install required.

**A commercial decision-support system for a (fictional) UAE fleet dealer group — not an academic ML exercise.**
Lead Scoring → Sales Pipeline → Conversion → Customer Churn → Revenue at Risk → Fleet Renewal/Upgrade → Marketing Analytics.

> ⚠️ **ALL DATA IS SYNTHETIC.** Generated with `numpy` + `Faker` under seed 42. No real company, person, contract or transaction is represented. Statistical relationships were engineered to mirror real B2B fleet-sales behaviour so the analytics are meaningful, but this is not market evidence. Currency **AED**. Analysis as-of date **30 June 2026**.

---

## 1. The business problem

A UAE dealer group sells vans, pickups, heavy trucks, earth-moving equipment, buses and passenger fleets to corporate accounts across seven emirates. Three things are broken:

1. **Sales effort is spread evenly across every lead**, so reps spend the same hours on a 2-unit enquiry with no budget as on a 40-unit fleet tender.
2. **Renewals are reactive.** Contracts expire and the customer has already spoken to a competitor.
3. **Churn is discovered after the fact.** There is no forward view of which recurring revenue is about to disappear, or what it is worth.

This system answers four questions for every analysis: **What happened? Why? Why does it matter? What should management do?**

---

## 2. Run it

```bash
pip install -r requirements.txt

python scripts/01_generate_data.py      # synthetic datasets
python scripts/02_eda_validation.py     # data quality gate -> data/data_quality_log.csv
python scripts/03_lead_scoring.py       # logistic regression + random forest
python scripts/04_churn_prediction.py   # churn model + revenue at risk
python scripts/05_analytics.py          # pipeline, marketing, fleet lifecycle
python scripts/06_export_excel.py       # builds Automotive_Sales_Engine.xlsx

streamlit run app.py                    # interactive dashboard, 6 pages
```

All paths are relative to the project root, so the folder can be moved or zipped freely.

---

## 3. Project structure

```
automotive_sales_engine/
├── README.md                        methodology, insights, actions (this file)
├── requirements.txt
├── Automotive_Sales_Engine.xlsx     15-sheet workbook, the primary deliverable
├── Top_50_Priority_Leads.csv        today's calling list
├── app.py                           Streamlit + Plotly dashboard
├── data/
│   ├── leads.csv          2,000 CRM leads
│   ├── accounts.csv         720 contracted customers
│   ├── activities.csv    11,397 sales touchpoints
│   ├── service.csv        4,248 service / warranty tickets
│   ├── leads_scored.csv           leads + Lead_Score, Lead_Priority, Recommended_Action
│   ├── accounts_scored.csv        accounts + Churn_Probability, Risk_Category, Revenue_at_Risk
│   ├── analytics_*.csv            funnel, pipeline, marketing, revenue-at-risk, fleet opportunities
│   └── data_quality_log.csv       every validation check, with severity
├── models/
│   ├── lead_scoring_model.pkl     fitted sklearn pipeline
│   ├── churn_model.pkl            fitted calibrated pipeline
│   └── model_metadata.json        real metrics, features, leakage controls, train date
└── scripts/                        01 → 06, reproducible end to end
```

---

## 4. Data model

| Table | Rows | Grain | Key |
|---|---|---|---|
| Leads | 2,000 | one CRM opportunity | `Lead_ID` |
| Accounts | 720 | one contracted customer | `Account_ID` |
| Activities | 11,397 | one sales touchpoint | `Activity_ID` → `Lead_ID` / `Account_ID` |
| Service | 4,248 | one service / warranty ticket | `Service_ID` → `Account_ID` |

The data is generated from a **latent-driver model**, not random noise: budget approval, engagement depth, lead source quality, rep skill, industry and deal size drive a true conversion propensity; service satisfaction, repeat failures, competitor presence, purchase recency and contract age drive a true churn propensity. The models then have to *recover* those relationships from observable columns — which is what makes the measured performance meaningful.

### Data quality (`02_eda_validation.py`)
Duplicate keys, unexpected nulls, referential integrity, logical consistency (no conversion before creation, no expiry before start, CSAT within 1–5), IQR outliers and class balance are all checked. **Issues are logged, never silently dropped** — the 225 high-value deal outliers are genuine enterprise fleet orders and are retained. Current status: **0 errors**.

---

## 5. Models — real scikit-learn outputs

| | Lead Scoring | Churn |
|---|---|---|
| Candidates | Logistic Regression, Random Forest | Logistic Regression, Random Forest |
| Champion | **Logistic Regression** | **Random Forest** (isotonic-calibrated) |
| Test ROC-AUC | **0.824** | **0.737** |
| 5-fold CV ROC-AUC | 0.804 ± 0.023 | 0.759 ± 0.035 |
| Accuracy / F1 | 0.758 / 0.699 | 0.683 / 0.529 |
| Precision / Recall | 0.655 / 0.750 | 0.478 / 0.593 |

*(Exact figures are regenerated on every run and written to `models/model_metadata.json` and the workbook's Model Performance sheet. Nothing is hand-entered.)*

**Leakage controls**
- Lead scoring **trains only on closed leads (Won/Lost)**. An open deal has no realised outcome; labelling it `0` would teach the model that "still in progress" means "will not buy". Open leads are scored, never trained on.
- `Sales_Stage`, `Conversion_Date`, `Sales_Cycle_Days`, `Lost_Reason` are excluded from features — all are known only after the outcome.
- Churn features are **recomputed from the service transaction table**, not copied from a pre-aggregated account field; `Churn_Status` is the target only.
- 75/25 stratified hold-out + 5-fold stratified CV. **Train and test AUC are both reported** so over-fitting is visible rather than hidden.
- The churn operating threshold is tuned on the **training fold only**.

**Why churn probabilities are calibrated.** Revenue at Risk is a money figure, so the probability must be a true expectation rather than a ranking score. Isotonic calibration (5-fold) means a 40% score really does mean ~40 of every 100 such accounts leave — otherwise `Annual_Spend × Probability` would be systematically wrong.

---

## 6. Business logic

| Measure | Definition |
|---|---|
| `Lead_Priority` | Very High >65 · High 46–65 · Medium 26–45 · Low ≤25 |
| `Risk_Category` | Critical: churn >70% **and** spend >AED 200k · High: >50% **and** >AED 100k · Medium: 30–50% · Low: <30% |
| **Revenue at Risk** | `Annual_Spend_AED × Churn_Probability / 100` — a live formula in the workbook |
| Weighted pipeline | `Deal_Value × stage win probability` (Prospect 10%, Qualified 25%, Proposal 45%, Negotiation 70% — editable blue cells) |
| Renewal opportunity | contract expiring within 90 days |
| Upgrade opportunity | fleet >10 units **and** no purchase in 3+ years |
| Cross-sell opportunity | spend up 50%+ year on year |
| `Urgency_Score` | 40% expiry proximity + 35% account value + 25% churn probability |

**Funnel assumption (documented).** A lead at Proposal has demonstrably passed Prospect and Qualified. For a *lost* deal the furthest stage reached is inferred from the loss reason: price/competitor → Negotiation, spec mismatch → Proposal, timing deferred → Qualified, no decision → Prospect. Without this, lost deals would vanish from the funnel and every conversion rate would be overstated.

**Marketing ROI caveat.** Cost per lead uses the prescribed nominal rates (LinkedIn $50, Trade Show $200, Website $20, Referral $0, Email $15, Partner $30, Outbound $40) and ROI is computed on a 12% assumed gross margin. Because those are digital-marketing rates while the deals are multi-million-AED fleet orders, **absolute ROI percentages are implausibly large — use them to rank channels against each other, not as a budget forecast.** Referral has no media cost, so its ROI is undefined; judge it on conversion and deal size.

---

## 7. Deliverables

**`Automotive_Sales_Engine.xlsx`** — 15 sheets: Read Me · Leads Raw · Accounts Raw · Activities Raw · Service Raw · Data Dictionary · Lead Scoring · Top 50 Leads · Churn Prediction · Revenue at Risk · Fleet Opportunities · Pipeline Analytics · Marketing Analytics · Model Performance · Executive Dashboard.
732 live formulas, 0 formula errors (verified by recalculation). Excel Tables with filter dropdowns on every dataset, conditional formatting on all risk and priority columns, 15 charts, and an Executive Dashboard whose KPI cards are `SUMIF`/`COUNTIF`/`SUMIFS` formulas reading the raw sheets — edit the data and every KPI moves.

**`app.py`** — six-page Streamlit + Plotly dashboard: Executive Overview · Lead Intelligence · Pipeline Analytics · Customer Churn & Risk · Fleet Opportunities · Marketing Analytics. Sidebar filters for date range, industry, emirate, city, salesperson, lead source, product and risk level cascade through every chart and table. An **account drill-down** on every page shows profile → fleet → spend → contract expiry → churn probability → revenue at risk → service health → recommended action.

---

## 8. Top business insights

1. **Half of all late-stage value dies at the closing table.** Prospect→Qualified→Proposal→Negotiation holds 83–90% at each step, then Negotiation→Won collapses to **46.6% (53.4% drop-off)**. → *The bottleneck is closing, not lead generation. Deal-desk support, approval authority on discounts, and a competitive-response playbook are worth more than more leads.*
2. **Price and competitors cause 57% of lost revenue** — Price/Budget AED 783M (30.0%) and Competitor Won AED 711M (27.2%) of AED 2.61B lost. → *Pricing flexibility and battlecards attack the majority of the loss pool.*
3. **AED 546.7M of recurring revenue is at risk — 38% of the AED 1.43B active book.** 158 accounts sit in Critical or High risk. → *Retention is currently worth more than acquisition: the weighted new-business pipeline is AED 455M, less than the exposure.*
4. **Exposure is concentrated: the top 20 accounts carry AED 121.5M, 22% of all revenue at risk; the top 50 carry AED 217M (40%).** → *Twenty named retention plans protect a fifth of the exposure. Spreading effort evenly across 720 accounts wastes it.*
5. **A competitor in the account more than doubles churn — 44.6% vs 20.6%**, across AED 666M of annual spend. → *Competitor presence should be a mandatory CRM field and trigger an automatic defence play.*
6. **Service quality is the churn lever.** Accounts with CSAT ≤3.5 churn at **50%**; above 4.5 they churn at **13%**. Purchase recency is the single strongest model feature, followed by average satisfaction and ticket volume. → *Every day of unresolved service is a retention cost, not an operations cost.*
7. **Referrals convert at 57.0% and Partners at 46.5%, versus LinkedIn 25.8% and Outbound 12.5%** — yet LinkedIn supplies the largest volume (496 leads) and Outbound 184. → *Reallocate outbound capacity into a formal referral and partner programme: same headcount, roughly double the conversion.*
8. **The two highest-value channels are the two worst converters.** Outbound carries the largest average deal (AED 5.61M) but converts at just 12.5%; Website is second largest (AED 5.28M) at 21.7%. Referrals convert at 57% on a smaller AED 3.52M average. → *These are big fleet enquiries being handled like small ones. Route high-value inbound and outbound enquiries to senior AEs on arrival instead of into a generic nurture queue — the value is already there, the conversion is not.*
9. **AED 125.9M of fleet lifecycle opportunity is sitting unworked**: 51 renewals (AED 101.4M of annual spend expiring within 90 days), 60 upgrade candidates (AED 78.0M potential), 72 cross-sell accounts growing 50%+ YoY.
10. **Rep performance varies almost 2× on win rate — 44.8% (Rashid Al Mansoori) to 29.4% (Ahmed Siddiqui)** — while Ahmed carries the *largest* average deal size (AED 5.4M). → *He is losing the biggest deals. Pair him with the top closer on large tenders; the coaching upside is the highest single lever in the sales team.*
11. **The top 50 scored leads hold AED 268.9M — 28.8% of all open pipeline — at an average score of 80/100**, with an expected (probability-weighted) value of AED 215M.

---

## 9. Recommended management actions

| # | Action | Owner | Value at stake |
|---|---|---|---|
| 1 | Work the **Top 50 leads** first (avg score 80, expected AED 215M). Very-High priority = contact within 24h. | Sales Director | AED 268.9M pipeline |
| 2 | Assign named retention plans to the **top 20 revenue-at-risk accounts**; commercial-director call within 7 days for the 47 Critical accounts. | Commercial Director | AED 121.5M |
| 3 | Open **renewal negotiations on the 51 contracts expiring within 90 days** now, before the competitor does. Multi-year lock-in with a service SLA. | Account Management | AED 101.4M annual spend |
| 4 | Fix the **Negotiation→Won stage**: deal desk, pre-approved discount bands, competitor battlecards. A 5-point improvement on a 53.4% drop-off is worth roughly AED 130M of currently-lost value. | Sales Ops | AED 2.61B lost pool |
| 5 | Launch the **fleet upgrade campaign** on 60 accounts with 3+ year-old fleets, leading with total-cost-of-ownership and residual value. | Marketing + Sales | AED 78.0M |
| 6 | **Shift outbound capacity into referral and partner programmes** (12.5% → 46–57% conversion) and route website enquiries straight to senior AEs. | Marketing | ~2× conversion on reallocated effort |
| 7 | Make **service recovery a retention KPI**: any account below CSAT 3.5 or with a repeat-issue rate above 25% gets an automatic recovery plan. | Aftersales | 50% → 13% churn spread |
| 8 | **Coach large-deal closing** — pair the lowest win-rate rep (29.4%, largest average deal) with the top closer (44.8%) on tenders above AED 5M. | Sales Director | ~AED 100M+ of at-risk large deals |

---

## 10. Interview narrative (2 minutes)

**Problem.** A UAE fleet dealer group was working every lead with equal effort, handling renewals reactively, and only learning about churn after the customer had gone. Management could not say which leads would convert, which customers were about to leave, or what that was worth in money.

**Data.** I built four linked synthetic datasets that mirror a real dealership CRM — 2,000 leads, 720 contracted accounts, 11,397 sales activities and 4,248 service tickets — generated from a latent-driver model so that budget approval, engagement, service quality and competitor presence genuinely drive outcomes rather than being random.

**Analytics.** Logistic Regression and Random Forest for lead scoring (champion ROC-AUC 0.824, trained only on closed deals to avoid labelling open pipeline as failure), and a calibrated Random Forest for churn (ROC-AUC 0.737) — calibrated deliberately, because Revenue at Risk is a money figure and needs a true probability, not a ranking score. On top of that sits deterministic commercial logic: revenue at risk, funnel drop-off, weighted pipeline, fleet lifecycle triggers and channel ROI.

**Insight.** AED 546.7M — 38% of the recurring book — is at risk, and 22% of it sits in just 20 accounts. The funnel holds 83–90% at every stage until Negotiation→Won, where it loses 53%. A competitor in the account doubles churn; CSAT below 3.5 quadruples it. Referrals convert at 57% while outbound manages 12.5%.

**Action.** Twenty named retention plans, renewal conversations on 51 expiring contracts, a deal desk aimed squarely at the closing stage, a 60-account fleet upgrade campaign, and a shift of outbound capacity into referral and partner programmes.

**Impact.** The system converts a flat CRM export into a ranked worklist: AED 268.9M of prioritised pipeline, AED 121.5M of concentrated retention exposure, and AED 125.9M of lifecycle opportunity — all delivered as an Excel workbook a commercial director can filter and a Streamlit dashboard a sales team can run daily.

---

*Built as a portfolio demonstration of B2B sales analytics, applied ML and BI engineering. Synthetic data throughout.*
