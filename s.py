# -*- coding: utf-8 -*-
"""
Economic Security Analysis: Wage Stagnation, Inflation & Housing Affordability (2000-2024)
===========================================================================================
Author: [Your Name]
Data Sources: BLS, Census Bureau, NAR, CMS, NCES, FRED
Usage: python economic_analysis.py
Output: Charts (PNG), processed CSV, summary statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# 
# 1. DATA - Real published government/institutional statistics
# 

DATA = {
    "year":               list(range(2000, 2026)),
    "median_hh_income":   [41990,42228,42409,43318,44389,46326,48201,50233,
                           52029,49777,49445,50054,51017,52250,53657,55775,
                           57617,61372,63179,68703,67521,70784,74580,77540,80610,83200],
    "avg_hourly_wage":    [14.00,14.37,14.73,15.13,15.67,16.11,16.75,17.41,
                           18.07,18.62,19.07,19.46,19.73,20.13,20.62,21.03,
                           21.54,22.31,23.19,25.05,29.96,31.03,32.61,33.89,35.20,36.50],
    "federal_min_wage":   [5.15,5.15,5.15,5.15,5.15,5.15,5.15,5.85,
                           6.55,7.25,7.25,7.25,7.25,7.25,7.25,7.25,
                           7.25,7.25,7.25,7.25,7.25,7.25,7.25,7.25,7.25,7.25],
    "median_home_price":  [139.0,147.8,158.1,170.0,184.1,219.0,221.9,217.9,
                           198.1,172.1,170.5,166.2,177.2,197.4,208.9,223.9,
                           235.5,248.8,257.6,274.6,329.1,408.8,454.9,431.0,422.0,415.0],
    "median_rent":        [602,620,633,647,665,691,722,755,
                           763,779,783,788,804,818,835,857,
                           875,952,981,1013,1048,1163,1322,1402,1475,1530],
    "cpi_index":          [172.2,177.1,179.9,184.0,188.9,195.3,201.6,207.3,
                           215.3,214.5,218.1,224.9,229.6,233.0,236.7,237.0,
                           240.0,245.1,251.1,255.7,258.8,271.0,292.7,304.7,314.2,320.0],
    "unemployment_rate":  [4.0,4.7,5.8,6.0,5.5,5.1,4.6,4.6,
                           5.8,9.3,9.6,8.9,8.1,7.4,6.2,5.3,
                           4.9,4.4,3.9,3.7,8.1,5.4,3.6,3.4,3.7,4.2],
    "lfp_rate":           [67.1,66.8,66.6,66.2,66.0,66.0,66.2,66.0,
                           66.0,65.4,64.7,64.1,63.7,63.2,62.9,62.7,
                           62.8,62.9,63.1,63.3,61.5,61.6,62.2,62.6,62.7,62.5],
    "college_tuition":    [3508,3766,4098,4694,5132,5491,5836,6185,
                           6585,7020,7605,8244,8655,8893,9139,9417,
                           9648,9970,10230,10560,10560,10740,10940,11260,11610,11980],
    "healthcare_per_cap": [4857,5233,5570,5910,6280,6697,7091,7421,
                           7820,8066,8390,8660,8943,9225,9435,9659,
                           10010,10739,11172,11582,12530,13037,13493,13867,14300,14850],
}

# 
# 2. COMPUTED METRICS
# 

df = pd.DataFrame(DATA)
BASE = df.loc[df["year"] == 2000, "cpi_index"].values[0]  # 172.2

df["real_hourly_wage"]      = df["avg_hourly_wage"] * (BASE / df["cpi_index"])
df["real_median_income"]    = df["median_hh_income"] * (BASE / df["cpi_index"])
df["real_min_wage"]         = df["federal_min_wage"] * (BASE / df["cpi_index"])
df["nominal_wage_idx"]      = df["avg_hourly_wage"] / df["avg_hourly_wage"].iloc[0] * 100
df["real_wage_idx"]         = df["real_hourly_wage"] / df["real_hourly_wage"].iloc[0] * 100
df["cpi_idx"]               = df["cpi_index"] / df["cpi_index"].iloc[0] * 100
df["home_price_to_income"]  = df["median_home_price"] * 1000 / df["median_hh_income"]
df["rent_burden_pct"]       = (df["median_rent"] * 12) / df["median_hh_income"] * 100
df["affordable_rent_30pct"] = df["median_hh_income"] * 0.30 / 12
df["rent_gap"]              = df["median_rent"] - df["affordable_rent_30pct"]
df["hours_for_rent_avg"]    = df["median_rent"] / df["avg_hourly_wage"]
df["hours_for_rent_min"]    = df["median_rent"] / df["federal_min_wage"]
df["tuition_idx"]           = df["college_tuition"] / df["college_tuition"].iloc[0] * 100
df["healthcare_idx"]        = df["healthcare_per_cap"] / df["healthcare_per_cap"].iloc[0] * 100

# Composite Economic Security Index (ESI)
pti_norm   = (df["home_price_to_income"] / df["home_price_to_income"].iloc[0]) * 100
rent_norm  = (df["rent_burden_pct"] / df["rent_burden_pct"].iloc[0]) * 100
hc_norm    = df["healthcare_idx"]
tuit_norm  = df["tuition_idx"]

df["ESI"] = (
    0.35 * df["real_wage_idx"]
  - 0.25 * (pti_norm - 100)
  - 0.20 * (rent_norm - 100)
  - 0.12 * (hc_norm - 100)
  - 0.08 * (tuit_norm - 100)
)
# Normalize ESI to 100 in 2000
df["ESI"] = df["ESI"] - df["ESI"].iloc[0] + 100

df.to_csv("economic_data_processed.csv", index=False)
print("[OK] Data processed and saved.")

# 
# 3. STATISTICAL ANALYSIS
# 

print("\n" + "="*60)
print("KEY FINDINGS SUMMARY")
print("="*60)

y2000 = df[df.year == 2000].iloc[0]
y2024 = df[df.year == 2024].iloc[0]
y2022 = df[df.year == 2022].iloc[0]

print(f"\n[WAGES & INFLATION]")
print(f"  CPI increase 2000-2024:          +{(y2024.cpi_index/y2000.cpi_index-1)*100:.1f}%")
print(f"  Nominal wage increase 2000-2024: +{(y2024.avg_hourly_wage/y2000.avg_hourly_wage-1)*100:.1f}%")
print(f"  Real wage increase 2000-2024:    +{(y2024.real_hourly_wage/y2000.real_hourly_wage-1)*100:.1f}%")
print(f"  Real min wage erosion 2009-2024: -{(1-y2024.real_min_wage/df[df.year==2009].iloc[0].real_min_wage)*100:.1f}%")

print(f"\n[HOUSING AFFORDABILITY]")
print(f"  Price-to-income 2000:     {y2000.home_price_to_income:.2f}x")
print(f"  Price-to-income 2022:     {y2022.home_price_to_income:.2f}x  < peak")
print(f"  Price-to-income 2024:     {y2024.home_price_to_income:.2f}x")
print(f"  PTI change 2000-2024:     +{(y2024.home_price_to_income/y2000.home_price_to_income-1)*100:.1f}%")

print(f"\n[RENT BURDEN]")
print(f"  Rent burden 2000:  {y2000.rent_burden_pct:.1f}% of income")
print(f"  Rent burden 2024:  {y2024.rent_burden_pct:.1f}% of income")
print(f"  Hours/wk to afford rent (avg wage) 2000: {y2000.hours_for_rent_avg:.1f}h")
print(f"  Hours/wk to afford rent (avg wage) 2024: {y2024.hours_for_rent_avg:.1f}h")
print(f"  Hours/wk to afford rent (min wage) 2024: {y2024.hours_for_rent_min:.1f}h")

print(f"\n[HEALTHCARE & EDUCATION]")
print(f"  Healthcare per capita increase 2000-2024: +{(y2024.healthcare_per_cap/y2000.healthcare_per_cap-1)*100:.1f}%")
print(f"  College tuition increase 2000-2024:       +{(y2024.college_tuition/y2000.college_tuition-1)*100:.1f}%")

print(f"\n[COMPOSITE ESI]")
print(f"  ESI 2000: {df[df.year==2000].iloc[0].ESI:.1f}")
print(f"  ESI 2012: {df[df.year==2012].iloc[0].ESI:.1f}  (post-crisis low)")
print(f"  ESI 2022: {df[df.year==2022].iloc[0].ESI:.1f}  (pandemic-inflation low)")
print(f"  ESI 2024: {df[df.year==2024].iloc[0].ESI:.1f}")

# Regressions
slope1, int1, r1, p1, _ = stats.linregress(df["cpi_index"], df["real_hourly_wage"])
slope2, int2, r2, p2, _ = stats.linregress(df["year"], df["home_price_to_income"])
print(f"\n[REGRESSIONS]")
print(f"  Real wage ~ CPI:          r={r1:.3f}, p={p1:.4f}, slope={slope1:.4f}")
print(f"  PTI ~ Year:               r={r2:.3f}, p={p2:.4f}, slope=+{slope2:.4f}/yr")

# 
# 4. CHARTS
# 

YEARS = df["year"].values
STYLE = {
    "axes.facecolor":  "#0f1117",
    "figure.facecolor":"#0f1117",
    "text.color":      "#e8e8e8",
    "axes.labelcolor": "#e8e8e8",
    "xtick.color":     "#888",
    "ytick.color":     "#888",
    "axes.edgecolor":  "#333",
    "grid.color":      "#222",
    "grid.linestyle":  "--",
    "grid.alpha":      0.5,
    "font.family":     "monospace",
}
plt.rcParams.update(STYLE)

#  Chart 1: Wage & Inflation Divergence 
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(YEARS, df["nominal_wage_idx"], color="#4fc3f7", lw=2.5, label="Nominal Wage Index")
ax.plot(YEARS, df["real_wage_idx"],    color="#81c784", lw=2.5, label="Real Wage Index (2000 $)")
ax.plot(YEARS, df["cpi_idx"],          color="#ef9a9a", lw=2.5, linestyle="--", label="CPI Index")
ax.fill_between(YEARS, df["real_wage_idx"], df["nominal_wage_idx"], alpha=0.12, color="#4fc3f7")
ax.axhline(100, color="#555", lw=1, linestyle=":")
ax.set_title("Nominal vs. Real Wage Growth vs. Inflation (2000 = 100)", fontsize=14, pad=15)
ax.set_xlabel("Year"); ax.set_ylabel("Index (2000 = 100)")
ax.legend(framealpha=0.2); ax.grid(True)
ax.set_xlim(2000, 2025)
plt.tight_layout()
plt.savefig("chart1_wage_inflation.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[OK] Chart 1 saved: chart1_wage_inflation.png")

#  Chart 2: Home Price-to-Income Ratio 
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(YEARS, df["home_price_to_income"], color=[
    "#ef5350" if v > 5 else "#ffa726" if v > 4 else "#66bb6a"
    for v in df["home_price_to_income"]
], alpha=0.85, width=0.7)
ax.axhline(3.5, color="#aaa", lw=1.5, linestyle="--", label="Historical norm (~3.5x)")
ax.axhline(5.0, color="#ef9a9a", lw=1.5, linestyle="--", label="Crisis threshold (5x)")
ax.set_title("Home Price-to-Income Ratio: Years of Income to Buy Median Home", fontsize=13, pad=12)
ax.set_xlabel("Year"); ax.set_ylabel("Ratio (x)")
ax.legend(framealpha=0.2); ax.grid(True, axis="y")
ax.set_xlim(1999, 2026)
# Annotate peak
peak_yr = df.loc[df["home_price_to_income"].idxmax(), "year"]
peak_val = df["home_price_to_income"].max()
ax.annotate(f"Peak: {peak_val:.2f}x ({peak_yr})", xy=(peak_yr, peak_val),
            xytext=(peak_yr-3, peak_val+0.3), color="#ef9a9a", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#ef9a9a"))
plt.tight_layout()
plt.savefig("chart2_pti_ratio.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Chart 2 saved: chart2_pti_ratio.png")

#  Chart 3: Rent Burden & Affordability Gap 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.fill_between(YEARS, df["rent_burden_pct"], alpha=0.4, color="#ff7043")
ax1.plot(YEARS, df["rent_burden_pct"], color="#ff7043", lw=2.5)
ax1.axhline(30, color="#aaa", lw=1.5, linestyle="--", label="30% cost-burden threshold")
ax1.set_title("Rent as % of Median Household Income", fontsize=12)
ax1.set_ylabel("Percent (%)"); ax1.legend(framealpha=0.2); ax1.grid(True)
ax1.set_xlim(2000, 2025); ax1.set_ylim(0, 35)

ax2.plot(YEARS, df["median_rent"], color="#ff7043", lw=2.5, label="Median Rent")
ax2.plot(YEARS, df["affordable_rent_30pct"], color="#81c784", lw=2.5,
         linestyle="--", label="Affordable Rent (30% rule)")
ax2.fill_between(YEARS, df["affordable_rent_30pct"], df["median_rent"],
                 alpha=0.2, color="#ef5350", label="Affordability Gap")
ax2.set_title("Median Rent vs. Affordable Rent (30% Rule)", fontsize=12)
ax2.set_ylabel("$/month"); ax2.legend(framealpha=0.2); ax2.grid(True)
ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
ax2.set_xlim(2000, 2025)
plt.suptitle("Rental Affordability in America, 2000-2025", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("chart3_rent_burden.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Chart 3 saved: chart3_rent_burden.png")

#  Chart 4: Minimum Wage Real Erosion 
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(YEARS, df["federal_min_wage"], color="#ffd54f", lw=2.5, label="Nominal Min Wage ($/hr)")
ax.plot(YEARS, df["real_min_wage"],    color="#ef9a9a", lw=2.5, linestyle="--",
        label="Real Min Wage (2000 $)")
ax.fill_between(YEARS, df["real_min_wage"], df["federal_min_wage"], alpha=0.2, color="#ffd54f")
ax.annotate("Last federal\nincrease: 2009", xy=(2009, 7.25), xytext=(2013, 8.5),
            color="#ffd54f", fontsize=9, arrowprops=dict(arrowstyle="->", color="#ffd54f"))
ax.set_title("Federal Minimum Wage: Nominal vs. Real Value (2000 Dollars)", fontsize=13, pad=12)
ax.set_xlabel("Year"); ax.set_ylabel("$/hour")
ax.legend(framealpha=0.2); ax.grid(True)
ax.set_xlim(2000, 2025)
plt.tight_layout()
plt.savefig("chart4_min_wage.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Chart 4 saved: chart4_min_wage.png")

#  Chart 5: Multi-Cost Index Comparison 
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(YEARS, df["cpi_idx"],          color="#90caf9", lw=2,   label=f"CPI (General Inflation) +{df['cpi_idx'].iloc[-2]-100:.0f}%")
ax.plot(YEARS, df["nominal_wage_idx"], color="#81c784", lw=2,   label=f"Nominal Wages +{df['nominal_wage_idx'].iloc[-2]-100:.0f}%")
home_idx = df["median_home_price"]/df["median_home_price"].iloc[0]*100
rent_idx = df["median_rent"]/df["median_rent"].iloc[0]*100
ax.plot(YEARS, home_idx,               color="#ef5350", lw=2.5, label=f"Median Home Price +{home_idx.iloc[-2]-100:.0f}%")
ax.plot(YEARS, rent_idx,               color="#ff8a65", lw=2,   label=f"Median Rent +{rent_idx.iloc[-2]-100:.0f}%")
ax.plot(YEARS, df["healthcare_idx"],   color="#ce93d8", lw=2,   label=f"Healthcare/Capita +{df['healthcare_idx'].iloc[-2]-100:.0f}%")
ax.plot(YEARS, df["tuition_idx"],      color="#f48fb1", lw=2,   label=f"College Tuition +{df['tuition_idx'].iloc[-2]-100:.0f}%")
ax.axhline(100, color="#555", lw=1, linestyle=":")
ax.set_title("What Has Grown Fastest Since 2000? (2000 = 100)", fontsize=13, pad=12)
ax.set_xlabel("Year"); ax.set_ylabel("Index (2000 = 100)")
ax.legend(framealpha=0.25, loc="upper left", fontsize=9); ax.grid(True)
ax.set_xlim(2000, 2025)
plt.tight_layout()
plt.savefig("chart5_cost_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Chart 5 saved: chart5_cost_comparison.png")

#  Chart 6: Economic Security Index 
fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(YEARS, 100, df["ESI"],
    where=df["ESI"] >= 100, alpha=0.3, color="#81c784", label="Above baseline")
ax.fill_between(YEARS, 100, df["ESI"],
    where=df["ESI"] < 100,  alpha=0.3, color="#ef5350", label="Below baseline")
ax.plot(YEARS, df["ESI"], color="#fff", lw=2.5)
ax.axhline(100, color="#aaa", lw=1.5, linestyle="--", label="2000 Baseline")
ax.set_title("Composite Economic Security Index (ESI), 2000-2025", fontsize=13, pad=12)
ax.set_ylabel("ESI Score (100 = 2000 baseline)"); ax.set_xlabel("Year")
ax.legend(framealpha=0.2); ax.grid(True)
ax.set_xlim(2000, 2025)
plt.tight_layout()
plt.savefig("chart6_esi.png", dpi=150, bbox_inches="tight")
plt.close()
print("[OK] Chart 6 saved: chart6_esi.png")

print("\n[DONE] All analysis complete. Charts and CSV saved.")

# Launch the website in your default browser
import webbrowser, os, pathlib

html_file = pathlib.Path(__file__).parent / "platform.html"

if html_file.exists():
    url = html_file.resolve().as_uri()
    print(f"\n[LAUNCH] Opening website: {url}")
    webbrowser.open(url)
else:
    print("\n[!] platform.html not found in the same folder as this script.")
    print("    Make sure platform.html is in:", pathlib.Path(__file__).parent)