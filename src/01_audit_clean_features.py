"""
01_audit_clean_features.py

Audit, clean, and feature-engineer the City Hotel subset.

Submission notes:
- The raw hotel_bookings.csv is included in this submission bundle at the user's request.
- Lisbon weather and Google Trends files are optional external inputs. The code supports them,
  but the supplied task files did not include them, so no fabricated values are used.
- KEEP_DUPLICATES=1 runs the duplicate-retention robustness variant and writes a separate
  city_hotel_clean_nodedup.csv without overwriting the main cleaned file.
"""
import glob
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)
KEEP_DUPLICATES = os.environ.get("KEEP_DUPLICATES") == "1"

MONTH_MAP = {m: i for i, m in enumerate([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
], 1)}

audit_lines = []
decisions = []

def log(msg):
    print(msg)
    audit_lines.append(msg)

def decide(step, finding, decision, rationale):
    decisions.append({"step": step, "finding": finding, "decision": decision, "rationale": rationale})

# 0. Load/filter
log("# City Hotel — Data Audit Report\n")
raw = pd.read_csv(os.path.join(RAW, "hotel_bookings.csv"))
log(f"Loaded full dataset: {raw.shape[0]:,} rows, {raw.shape[1]} columns ({raw.hotel.value_counts().to_dict()})")
df = raw[raw["hotel"] == "City Hotel"].copy()
log(f"Filtered to **City Hotel (H2 / Lisbon city hotel)**: {df.shape[0]:,} rows")
decide("Filter", "Source contains Resort Hotel (H1) and City Hotel (H2)", "Keep City Hotel rows only", "The assignment scope is the city hotel; H2 is the Lisbon property in the source dataset.")
raw_cancel_rate = df["is_canceled"].mean()
log(f"Raw City Hotel cancellation rate (before cleaning): {raw_cancel_rate:.3%}\n")

# 1. Missing values
log("## 1. Missing values\n")
miss = df.isna().sum(); miss = miss[miss > 0]
log(f"Columns with NaN: {miss.to_dict()}")
log(f"`agent`/`company` NaN are structural: agent={df.agent.isna().sum():,}, company={df.company.isna().sum():,}\n")
decide("Missing: children", "4 rows have children=NaN", "Impute 0", "Median/mode is 0 and the count is negligible.")
decide("Missing: country", "24 rows have country=NaN", "Fill with 'Unknown'", "Retains observations while making missingness explicit.")
decide("Missing: agent", f"{df.agent.isna().sum():,} rows have agent=NaN", "Create has_agent flag and keep raw ID only for audit", "NaN means no travel-agent identifier in this dataset.")
decide("Missing: company", f"{df.company.isna().sum():,} rows have company=NaN", "Create has_company flag; exclude raw company ID from modelling", "High-cardinality ID with structural missingness adds little beyond the flag.")
df["children"] = df["children"].fillna(0)
df["country"] = df["country"].fillna("Unknown")
df["has_agent"] = df["agent"].notna().astype(int)
df["has_company"] = df["company"].notna().astype(int)

# 2. Duplicates
log("## 2. Duplicate rows\n")
full_dup_mask = df.duplicated(keep="first")
n_dup = int(full_dup_mask.sum())
dup_rate_by_deposit = df.assign(dup=df.duplicated(keep=False)).groupby("deposit_type")["dup"].mean()
log(f"Exact full-row duplicates: {n_dup:,} rows (no booking-ID exists in the public dataset)")
log(f"Share of rows that are part of a duplicate group, by deposit type: {dup_rate_by_deposit.round(3).to_dict()}")
log(f"Non-Refund duplicate share = {dup_rate_by_deposit.get('Non Refund', 0):.1%}; No Deposit = {dup_rate_by_deposit.get('No Deposit', 0):.1%}.")
cancel_before = df["is_canceled"].mean()
df_dedup = df[~full_dup_mask].copy()
cancel_after = df_dedup["is_canceled"].mean()
log(f"Cancellation rate before dedup: {cancel_before:.3%}; after: {cancel_after:.3%}")
decide("Duplicates", f"{n_dup:,} exact full-row duplicates found", "Drop exact duplicates for the main analysis", "No booking ID exists to prove these are independent reservations; retaining them duplicates apparent events. A duplicate-retention robustness run is kept separately.")
df = df if KEEP_DUPLICATES else df_dedup

# 3. Dates
log("\n## 3. Date alignment\n")
df["arrival_date"] = pd.to_datetime(dict(
    year=df.arrival_date_year,
    month=df.arrival_date_month.map(MONTH_MAP),
    day=df.arrival_date_day_of_month
))
df["booking_date"] = df["arrival_date"] - pd.to_timedelta(df["lead_time"], unit="D")
log(f"Arrival dates span {df.arrival_date.min().date()} to {df.arrival_date.max().date()}")
log(f"Derived booking dates span {df.booking_date.min().date()} to {df.booking_date.max().date()}")
decide("Date derivation", "Raw file supplies arrival components and lead time, not booking date", "Reconstruct booking_date = arrival_date - lead_time", "This is the natural booking-time anchor available from the supplied schema.")

# 4. Outliers/invalid rows
log("\n## 4. Outliers and invalid rows\n")
zero_guests = (df.adults + df.children + df.babies) == 0
log(f"Rows with 0 total guests: {zero_guests.sum():,}")
decide("Outlier: zero guests", f"{zero_guests.sum():,} rows have no guests", "Drop", "No-guest observations are not meaningful customer stays for cancellation prediction.")
df = df[~zero_guests].copy()
extreme_mask = df.adr > 1000
log(f"Top ADR values include {df.adr.nlargest(3).tolist()}; rows with ADR > €1,000: {extreme_mask.sum():,}")
decide("Outlier: ADR", f"{extreme_mask.sum():,} row(s) have ADR > €1,000", "Drop the isolated extreme row", "The value is far beyond the rest of the distribution and is treated as a probable data-entry error.")
df = df[~extreme_mask].copy()
adr_zero = int((df.adr == 0).sum())
log(f"Rows with ADR=0: {adr_zero:,} — retained as complementary/free stays")
df["is_free_stay"] = (df.adr == 0).astype(int)
zero_nights = (df.stays_in_weekend_nights + df.stays_in_week_nights) == 0
log(f"Rows with 0 total nights: {zero_nights.sum():,} — retained and flagged")
df["is_zero_night"] = zero_nights.astype(int)
long_lead = int((df.lead_time > 365).sum())
log(f"Rows with lead_time > 365 days: {long_lead:,} (max={df.lead_time.max()} days) — retained")
decide("Lead time", f"{long_lead:,} bookings exceed one year lead time", "Keep without capping", "Long-horizon/group bookings can be genuine and contain cancellation signal.")

# 5. Leakage
log("\n## 5. Leakage controls\n")
leak_cols = ["reservation_status", "reservation_status_date"]
decide("Outcome leakage", "reservation_status and reservation_status_date directly encode outcome timing", "Drop both before modelling", "They are post-outcome fields and would make prediction circular.")
df = df.drop(columns=leak_cols)
log("Dropped outcome leakage fields: reservation_status, reservation_status_date.")
log("Near-arrival-only fields: assigned_room_type, booking_changes, room_type_changed, required_car_parking_spaces, total_of_special_requests.")
decide("Near-arrival-only fields", "Parking and special requests are accumulated in the booking lifecycle and are zero/absent for cancelled bookings", "Move required_car_parking_spaces and total_of_special_requests to the near-arrival model only", "Including them at booking time materially contaminates the target with information that emerges after booking creation.")

# 6. Holidays
log("\n## 6. Feature engineering: Portuguese public holidays\n")
holiday_file = os.path.join(RAW, "pt_holidays.csv")
if not os.path.exists(holiday_file):
    # Avoid an external package dependency in the submission. Build Portugal's national
    # public-holiday calendar for the covered years, including the movable Christian holidays
    # used by the original analysis (Good Friday, Easter Sunday, Corpus Christi).
    from datetime import date, timedelta
    def easter(y):
        a=y%19; b=y//100; c=y%100; d=b//4; e=b%4; f=(b+8)//25; g=(b-f+1)//3; h=(19*a+b-d-g+15)%30; i=c//4; k=c%4; l=(32+2*e+2*i-h-k)%7; m=(a+11*h+22*l)//451; month=(h+l-7*m+114)//31; day=((h+l-7*m+114)%31)+1; return date(y,month,day)
    years=range(int(df.arrival_date.dt.year.min()), int(df.arrival_date.dt.year.max())+1); dates=[]
    for y in years:
        e=easter(y); fixed=[date(y,1,1),date(y,4,25),date(y,5,1),date(y,6,10),date(y,8,15),date(y,10,5),date(y,11,1),date(y,12,1),date(y,12,8),date(y,12,25)]; dates.extend(fixed+[e-timedelta(days=2),e,e+timedelta(days=60)])
    pd.DataFrame({"date":pd.to_datetime(sorted(set(dates))) }).to_csv(holiday_file,index=False)
    log("Generated pt_holidays.csv from a self-contained Portugal national-holiday calendar for the covered years.")
hol = pd.read_csv(holiday_file, parse_dates=["date"])
holiday_dates = pd.DatetimeIndex(sorted(hol["date"]))
df["is_arrival_holiday"] = df["arrival_date"].isin(holiday_dates).astype(int)
def days_to_nearest_holiday(d):
    diffs = (holiday_dates - d).days
    return int(np.abs(diffs).min())
df["days_to_nearest_holiday"] = df["arrival_date"].map(days_to_nearest_holiday)
log(f"{df.is_arrival_holiday.sum():,} bookings arrive on a Portuguese public holiday; nearest-holiday distance median = {df.days_to_nearest_holiday.median():.0f} days.")

# 7. Optional weather
log("\n## 7. Optional arrival-date weather\n")
weather_path = os.path.join(RAW, "weather_lisbon.csv")
if os.path.exists(weather_path):
    with open(weather_path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    hdr = next(i for i, line in enumerate(lines) if line.lower().startswith(("time", "date")))
    wx = pd.read_csv(weather_path, skiprows=hdr)
    wx.columns = [c.split(" (")[0].strip() for c in wx.columns]
    wx = wx.rename(columns={"temperature_2m_max": "temp_max_c", "temperature_2m_min": "temp_min_c", "precipitation_sum": "precip_mm"})
    date_col = [c for c in wx.columns if "date" in c.lower() or "time" in c.lower()][0]
    wx[date_col] = pd.to_datetime(wx[date_col])
    keep = [c for c in ["temp_max_c", "temp_min_c", "precip_mm"] if c in wx.columns]
    wx = wx[[date_col] + keep].rename(columns={date_col: "arrival_date"})
    df = df.merge(wx, on="arrival_date", how="left")
    log(f"Joined {weather_path}; columns added: {keep}. Unmatched arrival dates: {int(df[keep].isna().any(axis=1).sum())}")
    decide("Feature: weather", "Historical Lisbon daily weather supplied", "Join on arrival_date and use only in the near-arrival model", "Actual arrival-day weather is not known with certainty at booking creation.")
else:
    log("**weather_lisbon.csv not supplied.** Weather features are not engineered and are not imputed.")
    decide("Feature: weather", "No historical weather file in supplied inputs", "Leave absent", "Do not fabricate external observations. The pipeline remains ready for Open-Meteo historical CSV export.")

# 8. Optional Google Trends
log("\n## 8. Optional Google Trends search interest\n")
trend_files = glob.glob(os.path.join(RAW, "trends_*.csv"))
if trend_files:
    days_since_sunday = (df["booking_date"].dt.dayofweek + 1) % 7
    df["booking_week"] = df["booking_date"].dt.normalize() - pd.to_timedelta(days_since_sunday, unit="D") - pd.Timedelta(days=7)
    trend_frames = []
    for f in trend_files:
        name = os.path.splitext(os.path.basename(f))[0].replace("trends_", "")
        with open(f, encoding="utf-8-sig") as fh:
            lines = fh.read().splitlines()
        hdr = next(i for i, line in enumerate(lines) if line.lower().startswith(("week", "date")))
        raw_t = pd.read_csv(f, skiprows=hdr, encoding="utf-8-sig")
        if raw_t.shape[1] < 2:
            continue
        raw_t = raw_t.iloc[:, :2].copy()
        raw_t.columns = ["week", name]
        raw_t["week"] = pd.to_datetime(raw_t["week"])
        raw_t[name] = pd.to_numeric(raw_t[name].astype(str).str.replace("<1", "0.5"), errors="coerce")
        trend_frames.append(raw_t.set_index("week")[name])
    if trend_frames:
        trends = pd.concat(trend_frames, axis=1).sort_index().reset_index().rename(columns={"week": "booking_week"})
        trends.columns = ["booking_week"] + [f"search_index_{c}" for c in trends.columns[1:]]
        df = df.merge(trends, on="booking_week", how="left")
        added = [c for c in trends.columns if c != "booking_week"]
        log(f"Joined {len(added)} Trends series using the last completed Sunday-started week before booking; unmatched rows: {int(df[added].isna().any(axis=1).sum())}.")
        decide("Feature: search interest", "Google Trends CSV supplied", "Use only the last completed week before booking", "Prevents look-ahead within the weekly search-index window.")
    else:
        log("Trend files were present but no usable two-column exports were found.")
else:
    log("**No trends_*.csv files supplied.** Search-interest features are not engineered and are not imputed.")
    decide("Feature: search interest", "No Google Trends export in supplied inputs", "Leave absent", "Do not fabricate indexed search interest. The pipeline supports standard Google Trends CSV exports.")

# 9. Additional engineered fields
log("\n## 9. Additional engineered fields\n")
df["booking_dow"] = df["booking_date"].dt.dayofweek
df["booking_month"] = df["booking_date"].dt.month
df["arrival_dow"] = df["arrival_date"].dt.dayofweek
df["arrival_month"] = df["arrival_date"].dt.month
df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
df["total_guests"] = df["adults"] + df["children"] + df["babies"]
df["room_type_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
log(f"Final cleaned + engineered City Hotel dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
log(f"Final cancellation rate: {df.is_canceled.mean():.3%}")

# 10. Feature dictionary
booking_time_features = {
    "lead_time": "Days between booking and arrival",
    "arrival_date_year": "Arrival year",
    "arrival_date_month": "Arrival month text",
    "arrival_date_week_number": "Arrival week number",
    "arrival_date_day_of_month": "Arrival day",
    "arrival_month": "Arrival month numeric",
    "arrival_dow": "Arrival day of week",
    "stays_in_weekend_nights": "Weekend nights booked",
    "stays_in_week_nights": "Week nights booked",
    "total_nights": "Total nights booked",
    "adults": "Adults on booking",
    "children": "Children on booking",
    "babies": "Babies on booking",
    "total_guests": "Total guests",
    "meal": "Meal plan selected",
    "country": "Guest country of origin — retained for prediction but monitored for fairness, not used as a recommended tariff",
    "market_segment": "Market segment",
    "distribution_channel": "Distribution channel",
    "is_repeated_guest": "Repeat guest flag",
    "previous_cancellations": "Past cancellations",
    "previous_bookings_not_canceled": "Past completed bookings",
    "reserved_room_type": "Room type requested at booking",
    "deposit_type": "Deposit type agreed at booking",
    "has_agent": "Travel-agent flag",
    "has_company": "Corporate booking flag",
    "days_in_waiting_list": "Waiting-list days",
    "customer_type": "Customer type",
    "adr": "Average daily rate agreed at booking",
    "is_free_stay": "Zero-rate stay flag",
    "is_zero_night": "Zero-night booking flag",
    "booking_dow": "Booking day of week",
    "booking_month": "Booking month",
    "is_arrival_holiday": "Arrival date is a PT holiday",
    "days_to_nearest_holiday": "Distance to nearest PT holiday",
}
for c in sorted([c for c in df.columns if c.startswith("search_index_")]):
    booking_time_features[c] = "Google Trends search interest for the last completed week before booking"
near_arrival_features = {
    **booking_time_features,
    "assigned_room_type": "Room type assigned later in the booking lifecycle",
    "booking_changes": "Cumulative booking changes by near arrival",
    "room_type_changed": "Reserved and assigned room type differ",
    "required_car_parking_spaces": "Parking spaces required; recorded during the stay lifecycle",
    "total_of_special_requests": "Special requests accumulated after booking",
}
for c in [c for c in ["temp_max_c", "temp_min_c", "precip_mm"] if c in df.columns]:
    near_arrival_features[c] = "Historical arrival-date weather; only near-arrival model"
dropped_leakage = {
    "reservation_status": "Outcome field — directly encodes cancellation/no-show",
    "reservation_status_date": "Date outcome was recorded",
}
rows = []
for name, desc in booking_time_features.items(): rows.append({"feature": name, "description": desc, "known_at": "known at booking time"})
for name, desc in near_arrival_features.items():
    if name not in booking_time_features: rows.append({"feature": name, "description": desc, "known_at": "known only near arrival"})
# explicitly include reclassified fields
for name in ["required_car_parking_spaces", "total_of_special_requests"]:
    rows.append({"feature": name, "description": near_arrival_features[name], "known_at": "known only near arrival"})
for name, desc in dropped_leakage.items(): rows.append({"feature": name, "description": desc, "known_at": "DROPPED (leakage)"})
# de-duplicate rows by feature, keeping strongest status
fdf = pd.DataFrame(rows).drop_duplicates(subset=["feature"], keep="last")

# Output files
if KEEP_DUPLICATES:
    path = os.path.join(OUT, "city_hotel_clean_nodedup.csv")
    df.to_csv(path, index=False)
    pd.DataFrame(decisions).to_csv(os.path.join(OUT, "cleaning_decisions_nodedup.csv"), index=False)
    with open(os.path.join(OUT, "audit_report_nodedup.md"), "w", encoding="utf-8") as f: f.write("\n".join(audit_lines))
    print(f"Robustness run (duplicates kept): wrote {os.path.basename(path)}")
    raise SystemExit(0)

df.to_csv(os.path.join(OUT, "city_hotel_clean.csv"), index=False)
fdf.to_csv(os.path.join(OUT, "feature_dictionary.csv"), index=False)
pd.DataFrame(decisions).to_csv(os.path.join(OUT, "cleaning_decisions.csv"), index=False)
with open(os.path.join(OUT, "audit_report.md"), "w", encoding="utf-8") as f: f.write("\n".join(audit_lines))

print("\nDone. Outputs written to data/processed/:")
for fn in ["city_hotel_clean.csv", "audit_report.md", "cleaning_decisions.csv", "feature_dictionary.csv", "pt_holidays.csv"]:
    p = os.path.join(OUT, fn) if fn != "pt_holidays.csv" else os.path.join(RAW, fn)
    if os.path.exists(p): print(" -", fn)
