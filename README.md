# City Hotel Cancellation, Overbooking & Deposit Policy — corrected

This repository is the canonical, corrected analysis package for the City Hotel (H2 / Lisbon) subset of the public Hotel Booking Demand dataset.

## What was fixed in this final pass
1. **Leakage correction.** `required_car_parking_spaces` and `total_of_special_requests` are excluded from the booking-time model and retained only in the near-arrival diagnostic. Outcome fields `reservation_status` and `reservation_status_date` are dropped.
2. **Literature and problem framing.** The report restores an eight-part problem identification and ten-source scholarly review.
3. **Deposit policy.** The former single uplift headline is withdrawn. Conservative/base/optimistic response assumptions are shown as a range, with a break-even recovery threshold.
4. **Overbooking.** Capacity and walk-cost assumptions are explicit, with sensitivity tables. The exact mathematical peak is reported separately from the conservative k=8 pilot point (smallest k within 1% of the base maximum).
5. **Data cleaning transparency.** The 41.8% raw City Hotel cancellation rate vs 30.1% after the main cleaning path is documented, and a duplicates-kept robustness branch is included.
6. **Fairness.** Country is a monitoring variable, not a direct deposit-pricing input. Subgroup calibration gaps are reported.
7. **External-data honesty.** Weather and Google Trends files were not supplied. No values are fabricated. Their loaders are retained and tested with fixture-format checks.
8. **Reproducibility and QA.** `src/03_model_policy_analysis.py` is the canonical analysis script. `src/run_analysis.py` is only the orchestration wrapper. Smoke tests now verify leakage guards and key outputs.

## Structure

- `src/01_audit_clean_features.py` — audit, cleaning, feature timing and optional external-data loaders
- `src/02_eda_charts.py` — descriptive charts
- `src/03_model_policy_analysis.py` — **canonical model + policy analysis**
- `src/03b_shap_only.py` — deterministic SHAP pass
- `src/04_robustness_and_fairness.py` — duplicate robustness and country monitoring tables
- `src/run_analysis.py` — thin orchestration wrapper
- `data/raw/hotel_bookings.csv` — user-supplied source data included because the final-submission instruction explicitly requested it
- `data/raw/pt_holidays.csv` — self-contained Portugal holiday calendar
- `data/processed/` — cleaned data and audit artefacts
- `outputs/` — corrected metrics, SHAP, fairness, overbooking and deposit outputs
- `outputs_nodedup/` — duplicates-kept robustness model outputs
- `tests/` — smoke tests covering structure and critical corrections
- `docs/MODEL_CARD.md` — model/policy governance card

## Reproduction

```bash
pip install -r requirements.txt
python src/run_analysis.py
pytest -q
```

The main pipeline needs only the supplied `hotel_bookings.csv` plus the included holiday calendar. Optional external files are:
- `data/raw/weather_lisbon.csv` — daily Open-Meteo-style historical weather export; loader handles metadata rows and unit-bearing column names.
- `data/raw/trends_<term>.csv` — standard Google Trends weekly export; the loader joins the last fully completed Sunday-started week before the booking date to avoid look-ahead.

Those external files were not supplied in this submission, so the final headline model does not use them.

## Temporal design

- Train: through 2016-12-12
- Calibration: 2016-12-13 to 2017-02-24
- Test: 2017-02-25 to 2017-08-31

## Corrected headline outputs

- Booking-time GBM ROC-AUC: **0.769**
- Near-arrival GBM ROC-AUC: **0.801**
- Base overbooking mathematical optimum: **k=18** at assumed 60 rooms and €200 walk cost
- Conservative pilot: **k=8**, the smallest base-cost level within 1% of maximum expected profit
- Deposit net incremental room-revenue range: **−€79.0k / +€54.7k / +€245.8k** (conservative/base/optimistic)
- Base break-even recovery fraction: **29.99%**

## Governance

Country is a fairness-monitoring attribute rather than a recommended deposit-pricing rule. Keep human override, log policy changes, version assumptions, validate capacity and current walk costs, and recalibrate on current data before production.

## Data provenance

Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets*. Data in Brief, 22, 41–49. DOI: 10.1016/j.dib.2018.11.126.

The source identifies H2 as the city hotel in Lisbon and covers 1 July 2015 to 31 August 2017. The raw source file is included here only because this submission package was explicitly requested to contain the data file; publication outside the assignment should follow the source licensing/usage terms.
