# City Hotel — Data Audit Report

Loaded full dataset: 119,390 rows, 32 columns ({'City Hotel': 79330, 'Resort Hotel': 40060})
Filtered to **City Hotel (H2 / Lisbon city hotel)**: 79,330 rows
Raw City Hotel cancellation rate (before cleaning): 41.727%

## 1. Missing values

Columns with NaN: {'children': 4, 'country': 24, 'agent': 8131, 'company': 75641}
`agent`/`company` NaN are structural: agent=8,131, company=75,641

## 2. Duplicate rows

Exact full-row duplicates: 25,902 rows (no booking-ID exists in the public dataset)
Share of rows that are part of a duplicate group, by deposit type: {'No Deposit': 0.287, 'Non Refund': 0.984, 'Refundable': 0.45}
Non-Refund duplicate share = 98.4%; No Deposit = 28.7%.
Cancellation rate before dedup: 41.727%; after: 30.039%

## 3. Date alignment

Arrival dates span 2015-07-01 to 2017-08-31
Derived booking dates span 2014-10-17 to 2017-08-30

## 4. Outliers and invalid rows

Rows with 0 total guests: 167
Top ADR values include [5400.0, 510.0, 451.5]; rows with ADR > €1,000: 1
Rows with ADR=0: 1,071 — retained as complementary/free stays
Rows with 0 total nights: 264 — retained and flagged
Rows with lead_time > 365 days: 2,703 (max=629 days) — retained

## 5. Leakage controls

Dropped outcome leakage fields: reservation_status, reservation_status_date.
Near-arrival-only fields: assigned_room_type, booking_changes, room_type_changed, required_car_parking_spaces, total_of_special_requests.

## 6. Feature engineering: Portuguese public holidays

2,939 bookings arrive on a Portuguese public holiday; nearest-holiday distance median = 10 days.

## 7. Optional arrival-date weather

**weather_lisbon.csv not supplied.** Weather features are not engineered and are not imputed.

## 8. Optional Google Trends search interest

**No trends_*.csv files supplied.** Search-interest features are not engineered and are not imputed.

## 9. Additional engineered fields

Final cleaned + engineered City Hotel dataset: 79,162 rows, 45 columns
Final cancellation rate: 41.785%