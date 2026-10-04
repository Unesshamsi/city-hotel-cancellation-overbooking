# Data

`hotel_bookings.csv` is the user-supplied source file and is included in the final submission package by explicit instruction.

The analysis filters to **City Hotel (H2 / Lisbon)**, removes exact full-row duplicates for the main path, drops invalid zero-guest rows and the isolated ADR > €1,000 row, and then creates time-aware booking and near-arrival feature sets.

### External-data status
- Portugal public holidays: included as `pt_holidays.csv` (self-contained calendar for the project period).
- Lisbon weather: **not supplied**. The optional loader is implemented but no values are fabricated.
- Google Trends: **not supplied**. The optional loader is implemented; its join uses the last fully completed Sunday-started week before the booking date to avoid look-ahead.
