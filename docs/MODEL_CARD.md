# City Hotel model card — corrected final version

## Model purpose
Estimate cancellation probability early enough to help revenue managers rank bookings for attention, then use the calibrated distribution in a separate overbooking simulation. A near-arrival model is maintained only as an operational diagnostic.

## Data
City Hotel (H2 / Lisbon) subset of the Hotel Booking Demand dataset. Main cleaned set: 53,273 rows. Historical period: Jul 2015–Aug 2017.

## Information timing
**Booking-time features** exclude `reservation_status`, `reservation_status_date`, `required_car_parking_spaces`, and `total_of_special_requests`. The latter two are retained only in the near-arrival diagnostic because their values can accumulate during the booking lifecycle.

**Near-arrival features** add assigned room type, booking changes, room-type change, parking and special requests.

## Evaluation
Strict chronological 70/15/15 split with calibration between training and final test. Final booking-time GBM ROC-AUC 0.769; near-arrival GBM ROC-AUC 0.801.

## Interpretation
SHAP provides contribution explanations, not causal effects. Country is monitored for fairness and calibration but is not recommended as a direct deposit-pricing rule.

## Policy use
Overbooking requires assumed room capacity and walk cost because inventory is absent. Deposit effects are scenario assumptions, not historical causal estimates. Final recommended pilot: k=8 under the 60-room assumption; deposit tiers 0% / 10% / 25% for <20% / 20–40% / >40% risk within eligible consumer-facing segments.

## Limitations
Historical data are old relative to deployment. Weather and Google Trends were absent. Capacity and walk costs are assumptions. Duplicate handling is judgement-based because no booking ID exists. Deposit response requires controlled experimentation.
