from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
import os, json, csv

ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
D=os.path.join(ROOT,'data','processed'); O=os.path.join(ROOT,'outputs'); C=os.path.join(ROOT,'charts'); OUT=os.path.join(ROOT,'docs')
meta=json.load(open(os.path.join(O,'run_metadata.json'),encoding='utf-8'))

def read_csv(path):
    with open(path,newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
metrics=read_csv(os.path.join(O,'model_metrics.csv'))
sh=read_csv(os.path.join(O,'shap_grouped_booking_time.csv'))
shn=read_csv(os.path.join(O,'shap_grouped_near_arrival.csv'))
dep=read_csv(os.path.join(O,'deposit_sensitivity.csv'))
dt=read_csv(os.path.join(O,'deposit_policy_tiers.csv'))
fair=read_csv(os.path.join(O,'fairness_by_country.csv'))
rob=read_csv(os.path.join(O,'robustness_dedup.csv'))
ob=read_csv(os.path.join(O,'overbooking_optima.csv'))
sens=read_csv(os.path.join(O,'overbooking_optimal_sensitivity.csv'))

clean_n=53273
clean_rate=0.3009779813
nonrefund_n=844
nonrefund_cancel=0.971564
raw_city_n=79330

def row(version, model='gradient_boosting'):
    for r in metrics:
        if r['version']==version and r['model']==model: return r
bt=row('booking_time'); na=row('near_arrival')
low=[x for x in dep if x['scenario']=='Conservative'][0]; base=[x for x in dep if x['scenario']=='Base'][0]; opt=[x for x in dep if x['scenario']=='Optimistic'][0]
ob_low=[x for x in ob if x['walk_scenario']=='Low'][0]; ob_base=[x for x in ob if x['walk_scenario']=='Base'][0]; ob_high=[x for x in ob if x['walk_scenario']=='High'][0]

fmt=lambda x,d=3: f"{float(x):.{d}f}"
eur=lambda x: f"€{float(x):,.0f}"
pct=lambda x: f"{float(x)*100:.1f}%"

refs=[
"Antonio, N., de Almeida, A., & Nunes, L. (2017). Predicting Hotel Bookings Cancellation with a Machine Learning Classification Model. IEEE ICMLA 2017. DOI: 10.1109/ICMLA.2017.00-11.",
"Antonio, N. (2019). Hotel booking demand and analytical decision support in hospitality. Tourism & Management Studies.",
"Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. Data in Brief, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126",
"Antonio, N., de Almeida, A., & Nunes, L. (2019). An automated machine learning based decision support system to predict hotel booking cancellations. Data Science Journal, 18(1), 32. https://doi.org/10.5334/dsj-2019-032",
"Chen, C.-C., Schwartz, Z., & Vargas, P. (2011). The search for the best deal: How hotel cancellation policies affect the search and booking decisions of deal-seeking customers. International Journal of Hospitality Management, 30(1), 129–135.",
"Liberman, V., & Yechiali, U. (1978). On the hotel overbooking problem: An inventory system with stochastic cancellations. Management Science, 24(11), 1117–1126. https://doi.org/10.1287/mnsc.24.11.1117",
"Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30.",
"Pan, B., Wu, D. C., & Song, H. (2012). Forecasting hotel room demand using search engine data. Journal of Hospitality and Tourism Technology, 3(3), 196–210. https://doi.org/10.1108/17579881211264486",
"Phumchusri, N., & Maneesophon, P. (2014). Optimal overbooking decision for hotel rooms revenue management. Journal of Hospitality and Tourism Technology, 5(3), 261–277. https://doi.org/10.1108/JHTT-03-2014-0006",
"Ramos Mir, V., Deyá Tortella, B., Leoni, V., & Nicolau, J. L. (2026). Decoding booking cancellations: Quantitative insights and theoretical advances in tourism behavior. Tourism Management, 114, 105384. https://doi.org/10.1016/j.tourman.2025.105384",
]

# DOCX setup
doc=Document(); sec=doc.sections[0]
sec.top_margin=Inches(.55); sec.bottom_margin=Inches(.50); sec.left_margin=Inches(.60); sec.right_margin=Inches(.60)
styles=doc.styles
styles['Normal'].font.name='Times New Roman'; styles['Normal'].font.size=Pt(9.2)
styles['Title'].font.name='Times New Roman'; styles['Title'].font.size=Pt(24)
styles['Heading 1'].font.name='Times New Roman'; styles['Heading 1'].font.size=Pt(15); styles['Heading 1'].font.bold=True
styles['Heading 2'].font.name='Times New Roman'; styles['Heading 2'].font.size=Pt(11); styles['Heading 2'].font.bold=True

for s in doc.sections:
    p=s.footer.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=p.add_run('City Hotel Decision Analytics · Final corrected submission'); r.font.name='Times New Roman'; r.font.size=Pt(7)

def P(text='', size=9.2, align='j'):
    q=doc.add_paragraph(); q.alignment={'j':WD_ALIGN_PARAGRAPH.JUSTIFY,'center':WD_ALIGN_PARAGRAPH.CENTER,'l':WD_ALIGN_PARAGRAPH.LEFT,'r':WD_ALIGN_PARAGRAPH.RIGHT}.get(align,WD_ALIGN_PARAGRAPH.JUSTIFY); q.paragraph_format.space_after=Pt(3); q.paragraph_format.line_spacing=1.0
    r=q.add_run(text); r.font.name='Times New Roman'; r.font.size=Pt(size); return q

def H(text, level=1):
    q=doc.add_heading(text, level=level); q.paragraph_format.space_before=Pt(3); q.paragraph_format.space_after=Pt(3); return q

def T(headers, rows, font=7.2):
    t=doc.add_table(rows=1,cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'
    for i,h in enumerate(headers):
        c=t.rows[0].cells[i]; c.text=str(h)
        for run in c.paragraphs[0].runs: run.font.name='Times New Roman'; run.font.size=Pt(font); run.bold=True
        c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    for rowv in rows:
        cells=t.add_row().cells
        for i,val in enumerate(rowv):
            cells[i].text=str(val); cells[i].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for pp in cells[i].paragraphs:
                pp.paragraph_format.space_after=Pt(0)
                for run in pp.runs: run.font.name='Times New Roman'; run.font.size=Pt(font)
    return t

def IMG(filename,width=6.5,cap=None):
    path=os.path.join(C,filename) if os.path.exists(os.path.join(C,filename)) else os.path.join(O,filename)
    if os.path.exists(path):
        q=doc.add_paragraph(); q.alignment=WD_ALIGN_PARAGRAPH.CENTER; q.add_run().add_picture(path,width=Inches(width)); q.paragraph_format.space_after=Pt(1)
        if cap: P(cap,7.2,'center')

def PB(): doc.add_page_break()

# Page 1
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(28)
r=p.add_run('CITY HOTEL'); r.bold=True; r.font.name='Times New Roman'; r.font.size=Pt(25)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('CANCELLATION, OVERBOOKING & DEPOSIT POLICY'); r.bold=True; r.font.name='Times New Roman'; r.font.size=Pt(17)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('Decision Analytics Report · Final corrected version'); r.italic=True; r.font.name='Times New Roman'; r.font.size=Pt(10.5)
P('Scope: City Hotel (H2 = the Lisbon city-hotel subset) of the Hotel Booking Demand dataset. The final pipeline uses a time-consistent booking-time information set; required_car_parking_spaces and total_of_special_requests are excluded from the booking-time model and retained only for near-arrival diagnostics.')
H('Executive summary')
P(f'After exact-duplicate removal and invalid-row screening, the analysis contains {clean_n:,} City Hotel bookings ({clean_rate*100:.1f}% cancelled). The source City Hotel subset is {raw_city_n:,} rows; the cleaning step is therefore a material governance choice, not a cosmetic preprocessing step.')
P(f'The temporal split is train through {meta["split"]["train_end"]}, calibration through {meta["split"]["calibration_end"]}, and final test from {meta["split"]["test_start"]}. The corrected booking-time GBM scores AUC {fmt(bt["roc_auc"])} / AP {fmt(bt["average_precision"])} with Brier {fmt(bt["brier"])} and ECE {fmt(bt["ece_10"])} on the untouched future test set. The near-arrival GBM reaches AUC {fmt(na["roc_auc"])} and is a late-stage diagnostic rather than the booking-time control.')
P(f'Overbooking uses explicit assumptions because room inventory is absent from the historical file: 60 rooms, median positive-rate room value of €{meta["overbooking"]["room_rate_eur"]:.2f}, and walk costs of €125/€200/€300. The mathematical expected-profit peak is k={meta["overbooking"]["base_optimal_k"]} at base cost, but the curve is deliberately treated as flat enough to recommend the smallest level within 1% of the maximum: k={meta["overbooking"]["recommended_k_within_1pct"]}.')
P(f'The deposit policy is not presented as a causal estimate. Under conservative/base/optimistic response assumptions, modeled net incremental room revenue is {eur(low["net_incremental_revenue_eur"])} / {eur(base["net_incremental_revenue_eur"])} / {eur(opt["net_incremental_revenue_eur"])}. The base case is positive only because the assumed response is strong enough; the break-even recovery fraction is {float(meta["deposit_policy"]["break_even_recovery_fraction"])*100:.1f}%.')
P('Recommendation: use the model as a governed revenue-manager co-pilot after current-data recalibration, capacity validation, fairness review and a controlled deposit experiment. Do not deploy the scenario uplift or overbooking level as an autonomous rule.')
PB()

# Page 2
H('1. Business problem and eight-part problem identification')
P('The business problem is a linked sequence. First, cancellation risk must be estimated at a clearly defined information timestamp. Second, those probabilities must feed a room-capacity decision under explicit capacity and walk-cost assumptions. Third, any deposit intervention must be judged against both cancellations prevented and bookings lost. The project therefore separates prediction, optimization and policy-response uncertainty.')
T(['#','Problem identification','Operational implication'],[
('1','Cancellation makes demand uncertain.','Use a calibrated early risk score, not a binary heuristic.'),
('2','Risk changes between booking and arrival.','Maintain separate booking-time and near-arrival information sets.'),
('3','Room inventory and walk cost are unobserved in the source file.','Show sensitivity rather than a single hidden optimum.'),
('4','Deposits can deter cancellation and reduce conversion.','Model response as scenarios; test before rollout.'),
('5','Channel/allotment mechanics can look like guest behaviour.','Keep process-heavy segments outside the consumer deposit lever.'),
('6','Country can shift predictions.','Use it for fairness monitoring, not direct deposit pricing.'),
('7','Weather and search data are absent.','Keep those features out of final scores; retain loaders for reproducibility.'),
('8','The source period is 2015–2017.','Treat results as backtest evidence; recalibrate on current data.')],7.1)
H('2. Literature review — evidence boundary')
P('The ten-source review below is intentionally selective. It supports cancellation prediction, explainable ML, search-based demand forecasting and stochastic overbooking. It does not provide an empirical basis for the project-specific 20%/35% deterrence or 4%/8% booking-loss parameters. Those remain assumptions.')
T(['Source','What it contributes','What it does not establish'],[
('Antonio et al. 2017','Hotel-cancellation prediction using machine learning.','No causal deposit-response estimate.'),
('Antonio 2019','Hospitality analytics / booking-demand framing.','Does not establish this hotel’s optimal capacity.'),
('Antonio et al. 2019, Data in Brief','Dataset provenance and field definitions.','Historical source; no current-data validation.'),
('Antonio et al. 2019, Data Science Journal','Automated cancellation decision support.','No direct deposit experiment.'),
('Chen, Schwartz & Vargas 2011','Cancellation policies can affect search/booking decisions.','Experimental setting; not this customer mix.'),
('Liberman & Yechiali 1978','Stochastic cancellation logic for hotel overbooking.','Requires assumptions about capacity and cancellation process.'),
('Lundberg & Lee 2017','SHAP-style model interpretation.','Explains contribution, not causality.'),
('Pan, Wu & Song 2012','Search-engine data can forecast hotel demand.','Demand forecasting is not cancellation prediction.'),
('Phumchusri & Maneesophon 2014','Optimal overbooking as a revenue-management problem.','Aggregate modelling; no deposit response.'),
('Ramos Mir et al. 2026','Recent behavioural evidence on booking cancellations.','Context is newer and different from the project dataset.')],6.7)
P('The review therefore motivates the architecture of the project, but the decision rules remain empirically local: score risk, quantify capacity assumptions, and experimentally validate any customer-facing intervention.')
PB()

# Page 3
H('3. Data, cleaning and temporal design')
T(['Item','Final treatment','Reason'],[
('Scope','City Hotel only: H2 / Lisbon.','Matches project decision scope.'),
('Exact duplicates','Removed for main analysis; retained in a robustness branch.','Avoid multiple counting of identical records; effect is reported.'),
('Zero guests','Dropped.','Invalid stay observation.'),
('ADR > €1,000','Single row removed.','Extreme isolated value inconsistent with distribution.'),
('ADR = 0','Kept and flagged.','Can represent complimentary stays.'),
('Outcome leakage','reservation_status and reservation_status_date dropped.','They encode the outcome / its recording date.'),
('Near-arrival fields','assigned_room_type, booking_changes, room_type_changed, parking, special requests.','Not defensibly fixed at booking creation.'),
('External weather/Trends','Not included because source files were not supplied.','No fabricated values; loaders remain in code.')],7.1)
H('4. What changed in the correction pass')
P('The primary correction is conceptual rather than cosmetic: required_car_parking_spaces and total_of_special_requests are no longer allowed into the booking-time model. In the cleaned data, every booking with at least one parking space was a completed stay, and cancellation falls as special requests accumulate. These are strong outcome-linked signals. Keeping them in the early model would therefore inflate performance and distort downstream policy decisions.')
T(['Statistic','Result'],[
('Raw City Hotel rows',f'{raw_city_n:,}'),('Clean main rows',f'{clean_n:,}'),('Raw/uncleaned cancellation rate','41.8%'),('Main cleaned cancellation rate','30.1%'),('Final test cancellation rate','28.2%'),('Non-Refund rows',f'{nonrefund_n:,}'),('Non-Refund cancellation rate',f'{nonrefund_cancel*100:.1f}%')],7.4)
P('The 41.8% → 30.1% change is discussed explicitly because it affects every rate, model split and downstream policy statistic. The Non-Refund result is also hedged: 844 rows is a small process-specific slice relative to the full 53,273-row cleaned population, and the project treats it as evidence of process mechanics, not a universal guest-behaviour law.')
H('5. Reproducibility status')
P('A self-contained Portugal-holiday calendar is included for the supplied period. The weather loader accepts a daily Open-Meteo-style export and the Trends loader accepts standard weekly Google Trends exports, joining only the last fully completed Sunday-started week before the booking date. Because those two files were not supplied, no weather or search-index values appear in the final model.')
IMG('01_audit_clean_features_flow.png',5.8,'Figure 1. Reproducible pipeline: raw booking data → audit/cleaning → EDA → time-split model → policy simulation → QA.')
PB()

# Page 4
H('6. Exploratory evidence')
IMG('01_cancel_by_lead_time.png',6.25,'Figure 2. Cancellation increases materially with longer lead time; bins are descriptive, not causal.')
IMG('02_cancel_by_deposit_type.png',5.9,'Figure 3. Deposit type is strongly associated with cancellation, but Non-Refund is treated as a process/channel effect.')
IMG('03_cancel_by_segment.png',5.9,'Figure 4. Market segment differences motivate explicit policy exclusions and monitoring.')
P('Seasonality is descriptive rather than a claim of monotonic change: the source period contains only Jul–Dec 2015 and Jan–Aug 2017, so month-by-month year comparisons are incomplete. Holiday proximity is used as a calendar feature, not as a causal explanation. The absence of external weather and search data means those potential drivers are documented limitations rather than silently imputed signals.')
PB()

# Page 5
H('7. Model performance and calibration')
T(['Model','Info set','AUC','AP','Brier','ECE10'],[
('Logistic regression','Booking time',fmt([r for r in metrics if r['version']=='booking_time' and r['model']=='logistic'][0]['roc_auc']),fmt([r for r in metrics if r['version']=='booking_time' and r['model']=='logistic'][0]['average_precision']),fmt([r for r in metrics if r['version']=='booking_time' and r['model']=='logistic'][0]['brier']),fmt([r for r in metrics if r['version']=='booking_time' and r['model']=='logistic'][0]['ece_10'])),
('Gradient boosting','Booking time',fmt(bt['roc_auc']),fmt(bt['average_precision']),fmt(bt['brier']),fmt(bt['ece_10'])),
('Logistic regression','Near arrival',fmt([r for r in metrics if r['version']=='near_arrival' and r['model']=='logistic'][0]['roc_auc']),fmt([r for r in metrics if r['version']=='near_arrival' and r['model']=='logistic'][0]['average_precision']),fmt([r for r in metrics if r['version']=='near_arrival' and r['model']=='logistic'][0]['brier']),fmt([r for r in metrics if r['version']=='near_arrival' and r['model']=='logistic'][0]['ece_10'])),
('Gradient boosting','Near arrival',fmt(na['roc_auc']),fmt(na['average_precision']),fmt(na['brier']),fmt(na['ece_10']))],7.2)
P(f'The booking-time GBM is the operational model: AUC {fmt(bt["roc_auc"])} on a future test set, with isotonic calibration chosen on the calibration period. The near-arrival GBM reaches AUC {fmt(na["roc_auc"])} but should not be confused with booking-time deployability; it contains fields whose values accumulate after booking creation.')
IMG('pr_curve_booking_time.png',5.85,'Figure 5. Precision–recall curves for the booking-time models.')
IMG('calibration_booking_time.png',5.85,'Figure 6. Out-of-time calibration for the booking-time models.')
P('The correction is visible in performance: the booking-time AUC is lower than the pre-correction 0.775 headline because two outcome-linked features were removed. That drop is a feature, not a failure: it is the cost of honest information timing.')
PB()

# Page 6
H('8. SHAP, fairness and governance')
P('SHAP is used as a model-explanation tool: it identifies features that move the prediction, but it does not prove that changing a feature will change cancellation.')
rows=[]
for r in sh[:10]: rows.append((r['feature'],f"{float(r['mean_abs_shap']):.3f}"))
T(['Booking-time feature','Mean |SHAP| (grouped)'],rows,7.3)
P('The leading booking-time drivers are country, market segment, lead time and ADR. Crucially, required_car_parking_spaces and total_of_special_requests are absent from this table because they are restricted to near-arrival scoring. The near-arrival SHAP table is retained in the repository for diagnostic use.')
T(['Country group','n','Mean predicted','Observed','Gap','High-risk share'],[
(r['country_group'],int(float(r['n'])),pct(r['mean_predicted']),pct(r['observed_cancel']),f"{float(r['calibration_gap'])*100:+.1f} pp",pct(r['share_high_risk'])) for r in fair],7.0)
P('Country is therefore a monitoring attribute rather than a recommended deposit-pricing rule. Portugal (PRT) shows a −9.5 percentage-point calibration gap in the test sample, while Great Britain (GBR) has a +5.4 point gap. The fairness table is an alerting mechanism: the appropriate response is threshold review, recalibration and subgroup validation, not demographic pricing.')
P('An ablation without country is also retained in outputs. It is a sensitivity check on how much predictive performance and risk ranking depend on a potentially policy-sensitive field; it is not a substitute for a formal fairness assessment.')
PB()

# Page 7
H('9. Overbooking optimization')
P(f'Because the source file has no room-inventory column, the overbooking model treats capacity as an explicit assumption. Base inputs are {int(meta["overbooking"]["base_capacity_rooms"])} rooms, €{meta["overbooking"]["room_rate_eur"]:.2f} median positive-rate room value, and walk costs of €125/€200/€300. Expected profit is computed from the calibrated booking-time show probabilities using a Poisson-binomial distribution.')
T(['Walk-cost scenario','Optimal k','Expected profit / night','Walk metric'],[
('Low €125',int(ob_low['overbooking_k']),eur(ob_low['avg_nightly_profit_eur']),'≈ 0.08 realised walks/night'),
('Base €200',int(ob_base['overbooking_k']),eur(ob_base['avg_nightly_profit_eur']),'≈ 0.07 realised walks/night'),
('High €300',int(ob_high['overbooking_k']),eur(ob_high['avg_nightly_profit_eur']),'≈ 0.05 realised walks/night')],7.4)
P(f'The exact base-cost maximum is k={meta["overbooking"]["base_optimal_k"]}. However, k={meta["overbooking"]["recommended_k_within_1pct"]} is the smallest level whose expected profit is within 1% of that peak, while the expected walk count is effectively zero in the backtest. This is the decision-grade point: a highly precise optimum is not warranted when capacity itself is assumed.')
IMG('profit_curve.png',6.15,'Figure 7. Expected nightly room profit under low/base/high walk-cost scenarios. The near-flat top supports a conservative pilot rather than a single-point “optimal” rule.')
T(['Assumed capacity','Low-cost k','Base-cost k','High-cost k'],[(r['capacity_rooms'],r['optimal_k'],[x['optimal_k'] for x in sens if x['capacity_rooms']==r['capacity_rooms'] and x['walk_scenario']=='Base'][0],[x['optimal_k'] for x in sens if x['capacity_rooms']==r['capacity_rooms'] and x['walk_scenario']=='High'][0]) for r in sens if r['walk_scenario']=='Low'],6.9)
P('Capacity sensitivity is strong: the mathematically preferred overbooking level changes from single digits at 40 rooms to much higher levels at 70–80 rooms. The project therefore requires verified inventory before any production deployment.')
PB()

# Page 8
H('10. Deposit policy: decision support, not causal proof')
T(['Risk tier','Eligible bookings','Deposit','Observed cancel','Assumed deterrence','Assumed booking loss'],[
(r['tier'],int(float(r['eligible_bookings'])),pct(r['deposit_pct']),pct(r['observed_cancel_rate']),pct(r['assumed_cancel_reduction']),pct(r['assumed_booking_loss_rate'])) for r in dt],7.1)
P('Policy exclusions: Non Refund deposit type, Groups market segment, and Offline TA/TO market segment. These are excluded because the historical relationships appear strongly intertwined with process/channel mechanics rather than representing a clean customer-facing price lever.')
T(['Scenario','Deterrence M/H','Booking loss M/H','Recovery of prevented value','Net incremental room revenue'],[
(r['scenario'],f"{float(r['medium_deterrence'])*100:.0f}% / {float(r['high_deterrence'])*100:.0f}%",f"{float(r['medium_booking_loss'])*100:.0f}% / {float(r['high_booking_loss'])*100:.0f}%",f"{float(r['revenue_recovery_fraction'])*100:.0f}%",eur(r['net_incremental_revenue_eur'])) for r in dep],7.0)
P(f'Base-case output is {eur(base["net_incremental_revenue_eur"])}; conservative is {eur(low["net_incremental_revenue_eur"])}; optimistic is {eur(opt["net_incremental_revenue_eur"])}. The base scenario prevents an expected {float(base["cancellations_prevented"]):.1f} cancellations while losing {float(base["bookings_lost"]):.1f} bookings under the assumed response. These figures are scenario math, not observed treatment effects.')
IMG('deposit_tradeoff.png',5.7,'Figure 8. Deposit trade-off: more cancellations prevented can require more bookings lost; policy value depends on the unknown response curve.')
P(f'Break-even insight: the modeled base policy needs about {float(meta["deposit_policy"]["break_even_recovery_fraction"])*100:.1f}% recovery of prevented room value to break even, holding the other assumptions fixed. The practical next step is an A/B or phased test with pre-registered cancellation and conversion metrics.')
PB()

# Page 9
H('11. Recommendation, controls and implementation')
T(['Decision','Recommendation','Control before go-live'],[
('Cancellation scoring','Use booking-time GBM as a ranked co-pilot.','Recalibrate on current bookings and monitor calibration drift.'),
('Overbooking','Pilot k=8 under the 60-room planning assumption.','Replace assumed capacity with live inventory and validate walk cost.'),
('Deposits','Test 10% on medium risk and 25% on high risk only within eligible segments.','Randomized/stepped experiment; measure conversion, cancellations and room value.'),
('Fairness','Country is monitoring-only.','Track subgroup calibration and high-risk share; require review for material gaps.'),
('External data','Add weather/Trends only when source files are supplied and validated.','Check time alignment and no look-ahead before release.'),
('Governance','Keep human override, version assumptions and log policy changes.','Retain model card, scenario register and test evidence.')],7.0)
H('12. 12-week change plan')
T(['Weeks','Workstream','Exit criterion'],[
('1–2','Current-data refresh + inventory confirmation','New time-split test set and verified room capacity.'),
('3–4','Model recalibration + subgroup audit','Stable calibration and documented fairness review.'),
('5–6','Overbooking shadow mode','Compare k=0, k=8 and live candidate levels without customer impact.'),
('7–9','Deposit experiment','Pre-registered response metrics by tier and eligible segment.'),
('10–11','Decision review','Economic result includes uncertainty and sensitivity.'),
('12','Governed release or rollback','Signed model card, assumptions register and monitoring owner.')],7.1)
H('Legal / responsible-use posture')
P('The model should be treated as decision support, not an autonomous pricing or access system. The project’s governance stance is consistent with the use of human override, documented policy changes, purpose limitation, data minimisation and monitoring of demographic/group-level effects. Legal review should be completed against the hotel’s actual jurisdiction, customer-notice obligations and current regulation before deployment.')
P('The source data are historical. The dashboard and model outputs are therefore a backtest. Production use requires fresh data, verified inventory, validated response rates and monitoring.')
PB()

# Page 10
H('13. Conclusion and limitations')
P(f'The corrected analysis supports one robust conclusion: cancellation risk is predictable enough to improve revenue-management attention, but the exact commercial rule should be conservative because the highest-value inputs are partly assumed. The booking-time GBM achieves AUC {fmt(bt["roc_auc"])} without the two outcome-linked features identified in the audit. Near-arrival information improves ranking, which is useful operationally but is not the same as early prediction.')
P(f'Overbooking should be framed as a capacity-sensitivity problem. Under the 60-room assumption, the mathematical peak is k={meta["overbooking"]["base_optimal_k"]}, but k={meta["overbooking"]["recommended_k_within_1pct"]} is a more defensible pilot point because it sits inside the flat top of the curve with effectively no realised walks in the backtest.')
P(f'Deposit policy should be framed as an experimental proposition. The corrected scenario range spans {eur(low["net_incremental_revenue_eur"])} to {eur(opt["net_incremental_revenue_eur"])} with {eur(base["net_incremental_revenue_eur"])} at the base assumptions. The previously presented single headline uplift is therefore withdrawn.')
H('Key limitations')
T(['Limitation','Why it matters','Mitigation'],[
('Historical 2015–2017 data','Current guest/channel behaviour may differ.','Recalibrate before production.'),
('No room-capacity field','Overbooking optimum is assumption-dependent.','Replace with live capacity and test.'),
('No causal deposit experiment','Scenario response parameters are invented.','Run controlled A/B or stepped experiment.'),
('Weather/Trends absent','Potential external signals are not assessed.','Load validated files and re-run.'),
('Duplicate removal requires judgement','No booking ID exists in the public source.','Report robustness branch.'),
('Country-related subgroup gaps','Calibration can differ by group.','Monitor and review; do not price directly on country.')],7.0)
H('References')
for ref in refs:
    P(ref,7.0)
H('Data / regulatory sources')
P('Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. Data in Brief, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126',7.0)
P('European Union. Regulation (EU) 2016/679 (General Data Protection Regulation).',7.0)
P('European Union. Regulation (EU) 2024/1689 (Artificial Intelligence Act).',7.0)
P('Open-Meteo. Historical Weather API documentation. Pipeline format only; values not supplied in this submission.',7.0)
P('Google Trends. Search-interest data/export documentation. Pipeline format only; values not supplied in this submission.',7.0)
P('Annexures: audit_report.md; cleaning_decisions.csv; feature_dictionary.csv; all model/policy CSVs and charts in outputs/; corrected dashboard, deck, video script, prompt logbook and repository ZIP in the submission folder.',7.0)

path=os.path.join(OUT,'City_Hotel_Full_Report.docx'); doc.save(path); print(path)
