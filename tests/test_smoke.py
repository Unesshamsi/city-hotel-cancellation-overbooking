from pathlib import Path
import csv, json


def test_project_structure_and_outputs():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "src/01_audit_clean_features.py",
        root / "src/02_eda_charts.py",
        root / "src/03_model_policy_analysis.py",
        root / "src/03b_shap_only.py",
        root / "src/04_robustness_and_fairness.py",
        root / "src/run_analysis.py",
        root / "data/raw/hotel_bookings.csv",
        root / "data/raw/pt_holidays.csv",
        root / "data/processed/city_hotel_clean.csv",
        root / "outputs/model_metrics.csv",
        root / "outputs/overbooking_optima.csv",
        root / "outputs/deposit_sensitivity.csv",
        root / "outputs/fairness_by_country.csv",
    ]
    for p in required:
        assert p.exists(), p


def test_booking_time_excludes_leakage():
    root = Path(__file__).resolve().parents[1]
    meta = json.loads((root / "outputs/run_metadata.json").read_text())
    bt = set(meta["features_booking_time"])
    assert "required_car_parking_spaces" not in bt
    assert "total_of_special_requests" not in bt
    assert "reservation_status" not in bt
    assert "reservation_status_date" not in bt


def test_near_arrival_contains_reclassified_fields():
    root = Path(__file__).resolve().parents[1]
    meta = json.loads((root / "outputs/run_metadata.json").read_text())
    near = set(meta["features_near_arrival"])
    assert {"required_car_parking_spaces", "total_of_special_requests"}.issubset(near)


def test_metrics_are_corrected_and_policy_is_scenario_based():
    root = Path(__file__).resolve().parents[1]
    with open(root / "outputs/model_metrics.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bt = next(r for r in rows if r["version"] == "booking_time" and r["model"] == "gradient_boosting")
    assert 0.75 < float(bt["roc_auc"]) < 0.79
    scenarios = {r["scenario"] for r in csv.DictReader(open(root / "outputs/deposit_sensitivity.csv", newline="", encoding="utf-8"))}
    assert scenarios == {"Conservative", "Base", "Optimistic"}
