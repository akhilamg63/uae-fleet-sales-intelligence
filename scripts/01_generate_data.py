"""
01_generate_data.py - Synthetic data generation
UAE B2B Automotive & Fleet Sales Intelligence Engine
ALL DATA IS SYNTHETIC. No real company, person or transaction is represented.
"""
import os, numpy as np, pandas as pd
from datetime import timedelta
from faker import Faker

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)
fake = Faker(); Faker.seed(SEED)

TODAY = pd.Timestamp("2026-06-30")   # analysis "as-of" date

INDUSTRIES = ["Logistics & Transport", "Construction & Contracting", "Car Rental & Leasing",
              "Oil & Gas Services", "Facilities Management", "Retail Distribution",
              "Government & Municipal", "Hospitality & Tourism"]
IND_W = [.20, .18, .14, .11, .11, .10, .08, .08]

SIZES = ["Small (<50)", "Medium (50-249)", "Large (250-999)", "Enterprise (1000+)"]
SIZE_W = [.30, .34, .24, .12]
SIZE_FLEET = {"Small (<50)": (3, 15), "Medium (50-249)": (10, 45),
              "Large (250-999)": (30, 140), "Enterprise (1000+)": (90, 400)}
SIZE_LIFT = {"Small (<50)": -.35, "Medium (50-249)": 0.0,
             "Large (250-999)": .35, "Enterprise (1000+)": .55}

SOURCES = ["LinkedIn", "Trade Shows", "Website", "Referral", "Email", "Partners", "Outbound"]
SRC_W = [.24, .12, .19, .11, .14, .09, .11]
SRC_COST = {"LinkedIn": 50, "Trade Shows": 200, "Website": 20, "Referral": 0,
            "Email": 15, "Partners": 30, "Outbound": 40}
SRC_QUALITY = {"Referral": 1.05, "Trade Shows": 0.80, "Partners": 0.55, "LinkedIn": 0.10,
               "Website": 0.00, "Email": -0.35, "Outbound": -0.55}

PRODUCTS = ["Light Commercial Vans", "Pickup Fleet", "Heavy Trucks", "Earth-Moving Equipment",
            "Passenger Sedan Fleet", "SUV Fleet", "Bus Fleet"]
PROD_W = [.20, .17, .15, .10, .14, .16, .08]
UNIT_PRICE = {"Light Commercial Vans": 95_000, "Pickup Fleet": 120_000, "Heavy Trucks": 320_000,
              "Earth-Moving Equipment": 470_000, "Passenger Sedan Fleet": 78_000,
              "SUV Fleet": 165_000, "Bus Fleet": 240_000}

EMIRATE_CITY = {"Dubai": ["Dubai Investment Park", "Al Quoz", "Jebel Ali", "Deira", "Dubai South"],
                "Abu Dhabi": ["Mussafah", "Al Ain", "Khalifa Industrial Zone", "Ruwais"],
                "Sharjah": ["Industrial Area", "Hamriyah Free Zone", "Al Sajaa"],
                "Ajman": ["Ajman Industrial", "Al Jurf"],
                "Ras Al Khaimah": ["Al Ghail", "RAK Maritime City"],
                "Fujairah": ["Fujairah Free Zone", "Dibba"],
                "Umm Al Quwain": ["UAQ Industrial Area"]}
EMIRATE_W = [.34, .25, .16, .08, .08, .05, .04]

SALES = {"SP-01": "Omar Al Farsi", "SP-02": "Priya Nair", "SP-03": "Hassan Khoury",
         "SP-04": "Elena Petrova", "SP-05": "Rashid Al Mansoori", "SP-06": "Daniel Okafor",
         "SP-07": "Mei Lin Tan", "SP-08": "Ahmed Siddiqui"}
SP_IDS = list(SALES)
SP_SKILL = dict(zip(SP_IDS, [0.35, 0.20, 0.05, -0.10, 0.28, -0.22, 0.12, -0.30]))

IND_LIFT = {"Logistics & Transport": .22, "Construction & Contracting": .16,
            "Car Rental & Leasing": .30, "Oil & Gas Services": .12,
            "Facilities Management": -.05, "Retail Distribution": -.10,
            "Government & Municipal": -.28, "Hospitality & Tourism": -.15}


def sigmoid(x): return 1 / (1 + np.exp(-x))


def pick(options, weights, n): return rng.choice(options, size=n, p=weights)


def company_names(n, prefix_seed=0):
    base = ["Gulf", "Emirates", "Al Noor", "Desert", "Falcon", "Arabian", "Oasis", "Zenith",
            "Horizon", "Nakheel", "Sahara", "Crescent", "Pearl", "Cedar", "Marina", "Summit",
            "Delta", "Vertex", "Anchor", "Meridian"]
    tail = ["Logistics LLC", "Contracting LLC", "Transport Co", "Rentals LLC", "Industries LLC",
            "Group", "Trading LLC", "Services FZE", "Holdings", "Equipment LLC", "Freight LLC",
            "Facilities LLC", "Motors LLC", "Fleet Solutions", "Enterprises"]
    out, seen, i = [], set(), prefix_seed
    while len(out) < n:
        nm = f"{base[i % len(base)]} {fake.last_name()} {tail[(i // 7) % len(tail)]}"
        if nm not in seen:
            seen.add(nm); out.append(nm)
        i += 1
    return out


# =================================================================== 1. LEADS
N_LEADS = 2000
print("Generating leads ...")
ind = pick(INDUSTRIES, IND_W, N_LEADS)
size = pick(SIZES, SIZE_W, N_LEADS)
src = pick(SOURCES, SRC_W, N_LEADS)
prod = pick(PRODUCTS, PROD_W, N_LEADS)
emirate = pick(list(EMIRATE_CITY), EMIRATE_W, N_LEADS)
city = np.array([rng.choice(EMIRATE_CITY[e]) for e in emirate])
sp = pick(SP_IDS, [1 / 8] * 8, N_LEADS)

fleet = np.array([rng.integers(*SIZE_FLEET[s]) for s in size])
units = np.clip((fleet * rng.uniform(0.15, 0.55, N_LEADS)).round().astype(int), 2, 45)
deal_value = np.array([units[i] * UNIT_PRICE[prod[i]] * rng.uniform(0.90, 1.12)
                       for i in range(N_LEADS)]).round(-3)

created = pd.to_datetime("2024-07-01") + pd.to_timedelta(rng.integers(0, 690, N_LEADS), unit="D")
days_in_pipe = np.clip((TODAY - created).days.values, 1, None)

budget_alloc = rng.random(N_LEADS) < sigmoid(
    0.6 + np.array([SRC_QUALITY[s] for s in src]) + np.array([SIZE_LIFT[s] for s in size]))

calls = np.clip(rng.poisson(np.clip(
    2.4 + 2.2 * budget_alloc + np.array([SIZE_LIFT[s] for s in size]) * 2
    + np.array([SP_SKILL[s] for s in sp]) * 2, 0.4, None)), 0, 18)

recency = np.clip(rng.integers(1, 160, N_LEADS) - calls * 3, 1, None)
recency = np.minimum(recency, days_in_pipe)
last_contact = TODAY - pd.to_timedelta(recency, unit="D")

lin = (-2.55
       + np.array([SRC_QUALITY[s] for s in src]) * 1.15
       + budget_alloc * 1.55
       + 0.085 * calls
       - 0.0065 * recency
       + np.log1p(deal_value / 1e6) * 0.42
       + np.array([SIZE_LIFT[s] for s in size])
       + np.array([SP_SKILL[s] for s in sp])
       + np.array([IND_LIFT[i] for i in ind])
       - 0.0016 * np.clip(days_in_pipe - 90, 0, None))
p_conv = sigmoid(lin + rng.normal(0, 0.45, N_LEADS))
converted = (rng.random(N_LEADS) < p_conv).astype(int)

stage = np.empty(N_LEADS, dtype=object)
for i in range(N_LEADS):
    if converted[i]:
        stage[i] = "Won"
    elif days_in_pipe[i] > 150:
        stage[i] = "Lost" if rng.random() < 0.80 else rng.choice(["Negotiation", "Proposal"])
    else:
        w = np.array([.30, .24, .16, .08, .22]) + np.array([-.18, -.02, .06, .10, .04]) * (p_conv[i] * 2)
        w = np.clip(w, .02, None); w = w / w.sum()
        stage[i] = rng.choice(["Prospect", "Qualified", "Proposal", "Negotiation", "Lost"], p=w)

cycle = np.clip(rng.normal(58, 22, N_LEADS) + fleet * 0.08, 18, 190).round().astype(int)
cycle_final = np.minimum(cycle, days_in_pipe)
conv_date = pd.Series([created[i] + timedelta(days=int(cycle_final[i])) if converted[i] else pd.NaT
                       for i in range(N_LEADS)])

lost_reason = np.where(stage == "Lost",
                       pick(["Price / Budget", "Competitor Won", "No Decision / Stalled",
                             "Timing Deferred", "Spec Mismatch"], [.30, .27, .21, .13, .09], N_LEADS), "")

leads = pd.DataFrame({
    "Lead_ID": [f"L{i:05d}" for i in range(1, N_LEADS + 1)],
    "Company_Name": company_names(N_LEADS),
    "Industry": ind, "Company_Size": size, "Fleet_Size": fleet,
    "Lead_Source": src, "Product_Interest": prod, "Units_Requested": units,
    "Budget_Allocated": np.where(budget_alloc, "Yes", "No"),
    "Budget_Range": pd.cut(deal_value, [-1, 5e5, 1.5e6, 4e6, 1e12],
                           labels=["< 500K", "500K - 1.5M", "1.5M - 4M", "4M+"]).astype(str),
    "Deal_Value_AED": deal_value,
    "Sales_Stage": stage,
    "Lead_Created_Date": created.date, "Last_Contact_Date": last_contact.date,
    "Days_in_Pipeline": days_in_pipe, "Number_of_Sales_Calls": calls,
    "Days_Since_Last_Contact": recency,
    "Emirate": emirate, "City": city,
    "Salesperson_ID": sp, "Salesperson": [SALES[s] for s in sp],
    "Sales_Cycle_Days": np.where(converted == 1, cycle_final, np.nan),
    "Converted_to_Sale": converted,
    "Conversion_Date": [d.date() if pd.notna(d) else None for d in conv_date],
    "Lost_Reason": lost_reason,
})

# ================================================================ 2. ACCOUNTS
N_ACC = 720
print("Generating accounts ...")
a_ind = pick(INDUSTRIES, IND_W, N_ACC)
a_size = pick(SIZES, [.22, .34, .28, .16], N_ACC)
a_fleet = np.array([rng.integers(*SIZE_FLEET[s]) for s in a_size])
a_em = pick(list(EMIRATE_CITY), EMIRATE_W, N_ACC)
a_city = np.array([rng.choice(EMIRATE_CITY[e]) for e in a_em])
a_prod = pick(PRODUCTS, PROD_W, N_ACC)
a_sp = pick(SP_IDS, [1 / 8] * 8, N_ACC)

c_start = pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.integers(0, 1700, N_ACC), unit="D")
term_yrs = rng.choice([2, 3, 4, 5], N_ACC, p=[.34, .34, .20, .12])
c_expiry = c_start + pd.to_timedelta((term_yrs * 365).astype(int), unit="D")
contract_age = ((TODAY - c_start).days.values / 365.25).round(2)
days_to_expiry = (c_expiry - TODAY).days.values

annual_spend = np.round(np.minimum(a_fleet, 110) * np.array([UNIT_PRICE[p] for p in a_prod]) *
                        rng.uniform(0.10, 0.34, N_ACC), -3)
spend_prev = np.round(annual_spend / np.clip(rng.normal(1.10, 0.30, N_ACC), 0.55, 2.4), -3)
spend_growth = ((annual_spend - spend_prev) / spend_prev * 100).round(1)

competitor = rng.random(N_ACC) < sigmoid(-0.55 + 0.35 * (annual_spend > 1.2e6))
days_since_purchase = np.clip(rng.gamma(2.3, 260, N_ACC), 12, 1900).astype(int)
last_purchase = TODAY - pd.to_timedelta(days_since_purchase, unit="D")

tickets = np.clip(rng.poisson(3.2 + a_fleet / 28), 0, 45)
sat = np.clip(4.35 - 0.055 * tickets - 0.35 * competitor + rng.normal(0, 0.35, N_ACC), 1.4, 5.0)
repeat_rate = np.clip(0.06 + 0.02 * tickets + rng.normal(0, 0.05, N_ACC), 0, 0.85)
resolution = np.clip(rng.normal(3.4, 1.4, N_ACC) + tickets * 0.09, 0.5, 18)
days_since_activity = np.clip(rng.gamma(2.2, 26, N_ACC), 2, 400).astype(int)

churn_lin = (-3.55 - 0.78 * (sat - 3.9) + 0.062 * tickets + 1.15 * competitor
             + 0.0019 * days_since_purchase + 0.0042 * days_since_activity
             + 1.35 * repeat_rate + 0.055 * resolution - 0.16 * contract_age
             + np.where(days_to_expiry < 120, 0.62, 0) - 0.30 * (annual_spend > 2e6))
churn = (rng.random(N_ACC) < sigmoid(churn_lin + rng.normal(0, 0.22, N_ACC))).astype(int)

accounts = pd.DataFrame({
    "Account_ID": [f"A{i:05d}" for i in range(1, N_ACC + 1)],
    "Company_Name": company_names(N_ACC, prefix_seed=5000),
    "Industry": a_ind, "Company_Size": a_size, "Fleet_Size": a_fleet,
    "Primary_Product": a_prod,
    "Annual_Spend_AED": annual_spend, "Prev_Year_Spend_AED": spend_prev,
    "Spend_Growth_Pct": spend_growth,
    "Contract_Start": c_start.date, "Contract_Term_Years": term_yrs,
    "Contract_Expiry": c_expiry.date, "Contract_Age_Years": contract_age,
    "Days_To_Expiry": days_to_expiry,
    "Last_Purchase_Date": last_purchase.date, "Days_Since_Last_Purchase": days_since_purchase,
    "Days_Since_Last_Activity": days_since_activity,
    "Service_Tickets_12M": tickets, "Avg_Satisfaction_Score": sat.round(2),
    "Repeat_Issue_Rate": repeat_rate.round(3), "Avg_Resolution_Days": resolution.round(1),
    "Competitor_Present": np.where(competitor, "Yes", "No"),
    "Emirate": a_em, "City": a_city,
    "Primary_Contact": [fake.name() for _ in range(N_ACC)],
    "Account_Manager_ID": a_sp, "Account_Manager": [SALES[s] for s in a_sp],
    "Account_Value_Tier": pd.cut(annual_spend, [-1, 6e5, 1.8e6, 4.5e6, 1e12],
                                 labels=["Bronze", "Silver", "Gold", "Platinum"]).astype(str),
    "Churn_Status": churn,
})

# ============================================================== 3. ACTIVITIES
print("Generating activities ...")
rows, aid = [], 1
for r in leads.itertuples():
    for _ in range(int(max(2, r.Number_of_Sales_Calls))):
        d = pd.to_datetime(r.Lead_Created_Date) + timedelta(days=int(rng.integers(0, max(2, r.Days_in_Pipeline))))
        t = rng.choice(["Call", "Email", "Meeting", "Proposal", "Demo"], p=[.36, .28, .18, .10, .08])
        rows.append((f"ACT{aid:06d}", "", r.Lead_ID, t, d.date(),
                     int(np.clip(rng.normal({"Call": 14, "Email": 5, "Meeting": 52, "Proposal": 40,
                                             "Demo": 75}[t], 9), 2, 180)),
                     rng.choice(["Positive", "Neutral", "Follow-up Required", "No Response"],
                                p=[.34, .28, .26, .12] if r.Converted_to_Sale else [.18, .28, .30, .24]),
                     r.Salesperson_ID, f"{t} logged for {r.Company_Name}"))
        aid += 1
for r in accounts.itertuples():
    for _ in range(int(rng.integers(2, 7))):
        d = TODAY - timedelta(days=int(rng.integers(1, 400)))
        t = rng.choice(["Call", "Email", "Meeting", "Proposal", "Demo"], p=[.34, .30, .22, .08, .06])
        rows.append((f"ACT{aid:06d}", r.Account_ID, "", t, d.date(),
                     int(np.clip(rng.normal(28, 12), 3, 180)),
                     rng.choice(["Positive", "Neutral", "Follow-up Required", "No Response"],
                                p=[.20, .30, .30, .20] if r.Churn_Status else [.42, .30, .21, .07]),
                     r.Account_Manager_ID, f"Account touchpoint - {t}"))
        aid += 1
activities = pd.DataFrame(rows, columns=["Activity_ID", "Account_ID", "Lead_ID", "Activity_Type",
                                         "Activity_Date", "Duration_Minutes", "Outcome",
                                         "Salesperson_ID", "Notes"])

# ================================================================= 4. SERVICE
print("Generating service records ...")
srows, sid = [], 1
for r in accounts.itertuples():
    for _ in range(int(max(1, r.Service_Tickets_12M))):
        d = TODAY - timedelta(days=int(rng.integers(1, 365)))
        srows.append((f"SR{sid:06d}", r.Account_ID, d.date(),
                      rng.choice(["Preventive", "Corrective", "Warranty", "Emergency"],
                                 p=[.40, .34, .16, .10]),
                      int(np.clip(round(rng.normal(r.Avg_Satisfaction_Score, 0.55)), 1, 5)),
                      rng.choice(["Mechanical Fault", "Scheduled Maintenance", "Parts Availability",
                                  "Warranty Claim", "Downtime / SLA Breach", "Telematics / Software"],
                                 p=[.26, .30, .14, .12, .10, .08]),
                      round(float(np.clip(rng.normal(r.Avg_Resolution_Days, 1.2), 0.2, 30)), 1),
                      "Yes" if rng.random() < r.Repeat_Issue_Rate else "No",
                      round(float(np.clip(rng.normal(4200, 2600), 250, 40000)), 2)))
        sid += 1
service = pd.DataFrame(srows, columns=["Service_ID", "Account_ID", "Service_Date", "Service_Type",
                                       "Satisfaction_Score", "Issue_Category",
                                       "Resolution_Time_Days", "Repeat_Issue", "Cost_AED"])

for name, df in [("leads", leads), ("accounts", accounts),
                 ("activities", activities), ("service", service)]:
    df.to_csv(os.path.join(DATA, f"{name}.csv"), index=False)
    print(f"  {name:12s} {len(df):>6,} rows x {df.shape[1]} cols")
print(f"Conversion rate: {leads.Converted_to_Sale.mean():.1%} | Churn rate: {accounts.Churn_Status.mean():.1%}")
print("Synthetic data generation complete. AS-OF DATE:", TODAY.date())
