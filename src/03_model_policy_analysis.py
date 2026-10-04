"""
03_model_policy_analysis.py

Canonical modelling and policy analysis script for the final submission.
Runs four models (logistic + LightGBM at booking time and near arrival), calibrates
probabilities on a chronological calibration window, produces SHAP explanations, and
simulates overbooking/deposit policies on the untouched future test window.

Critical leakage correction:
- required_car_parking_spaces and total_of_special_requests are NOT booking-time features.
- They are included only in the near-arrival diagnostic model.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve
from lightgbm import LGBMClassifier
from scipy.special import logit
import shap

HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.abspath(os.path.join(HERE,".."))
OUT=os.path.join(ROOT,"outputs"+os.environ.get("OUT_SUFFIX",""))
os.makedirs(OUT,exist_ok=True)
clean_file=os.environ.get("CLEAN_FILE","city_hotel_clean.csv")
df=pd.read_csv(os.path.join(ROOT,"data","processed",clean_file),parse_dates=["arrival_date","booking_date"])
y=df.is_canceled.astype(int)

base_features=[
    "lead_time","arrival_month","arrival_dow","stays_in_weekend_nights","stays_in_week_nights","total_nights",
    "adults","children","babies","total_guests","meal","country","market_segment","distribution_channel",
    "is_repeated_guest","previous_cancellations","previous_bookings_not_canceled","reserved_room_type",
    "deposit_type","has_agent","has_company","days_in_waiting_list","customer_type","adr","is_free_stay",
    "is_zero_night","booking_dow","booking_month","is_arrival_holiday","days_to_nearest_holiday"
]
# Optional search interest is genuinely available at booking time when supplied.
search_cols=[c for c in df.columns if c.startswith("search_index_")]
base_features += sorted(search_cols)
near_features=base_features+["assigned_room_type","booking_changes","room_type_changed","required_car_parking_spaces","total_of_special_requests"]
weather_cols=[c for c in ["temp_max_c","temp_min_c","precip_mm"] if c in df.columns]
near_features += weather_cols

# Temporal split: 70/15/15 by reconstructed booking date.
cut70=df.booking_date.quantile(.70); cut85=df.booking_date.quantile(.85)
tr=df.booking_date<=cut70; ca=(df.booking_date>cut70)&(df.booking_date<=cut85); te=df.booking_date>cut85

def preprocessor(features):
    cats=[c for c in features if df[c].dtype=="object"]
    nums=[c for c in features if c not in cats]
    return ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),nums),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",min_frequency=3))]),cats)
    ])

def calibrate_sigmoid(p,ycal,ptest):
    c=LogisticRegression(C=1e6,solver="lbfgs"); c.fit(logit(np.clip(p,1e-6,1-1e-6)).reshape(-1,1),ycal)
    return c.predict_proba(logit(np.clip(ptest,1e-6,1-1e-6)).reshape(-1,1))[:,1],c

def calibrate_isotonic(p,ycal,ptest):
    c=IsotonicRegression(out_of_bounds="clip"); c.fit(p,ycal); return c.predict(ptest),c

def ece(ytrue,pred,n_bins=10):
    bins=np.linspace(0,1,n_bins+1); ids=np.digitize(pred,bins[1:-1]); out=0; rows=[]
    for b in range(n_bins):
        m=ids==b
        if m.sum()==0: continue
        ap=float(pred[m].mean()); ay=float(np.mean(ytrue[m])); out+=float(m.mean())*abs(ap-ay)
        rows.append({"bin":b,"n":int(m.sum()),"predicted":ap,"observed":ay})
    return out,pd.DataFrame(rows)

models={}; metrics=[]
for version,features in [("booking_time",base_features),("near_arrival",near_features)]:
    X=df[features]; pre=preprocessor(features)
    lr=Pipeline([("pre",pre),("clf",LogisticRegression(C=1.0,max_iter=1200,solver="liblinear"))])
    gb=Pipeline([("pre",clone(pre)),("clf",LGBMClassifier(n_estimators=350,learning_rate=.035,num_leaves=31,subsample=.9,colsample_bytree=.85,reg_lambda=2.0,random_state=42,n_jobs=-1,verbosity=-1))])
    for name,model in [("logistic",lr),("gradient_boosting",gb)]:
        model.fit(X[tr],y[tr])
        pcal_raw=model.predict_proba(X[ca])[:,1]; ptest_raw=model.predict_proba(X[te])[:,1]
        # Calibrate each model using only calibration period; isotonic is used consistently for GBM.
        if name=="gradient_boosting": ptest,cal=calibrate_isotonic(pcal_raw,y[ca],ptest_raw); method="isotonic"
        else: ptest,cal=calibrate_sigmoid(pcal_raw,y[ca],ptest_raw); method="sigmoid"
        e,calbins=ece(y[te].to_numpy(),ptest)
        metrics.append({"version":version,"model":name,"calibration_method":method,"train_n":int(tr.sum()),"calibration_n":int(ca.sum()),"test_n":int(te.sum()),"test_cancel_rate":float(y[te].mean()),"roc_auc":roc_auc_score(y[te],ptest),"average_precision":average_precision_score(y[te],ptest),"brier":brier_score_loss(y[te],ptest),"ece_10":e})
        models[(version,name)]={"model":model,"calibrator":cal,"p_test":ptest,"p_test_raw":ptest_raw,"features":features,"calibration_method":method}
        pr_p,pr_r,_=precision_recall_curve(y[te],ptest); roc_fpr,roc_tpr,_=roc_curve(y[te],ptest); ctrue,cpred=calibration_curve(y[te],ptest,n_bins=10,strategy="quantile")
        pd.DataFrame({"recall":pr_r,"precision":pr_p}).to_csv(os.path.join(OUT,f"pr_{version}_{name}.csv"),index=False)
        pd.DataFrame({"fpr":roc_fpr,"tpr":roc_tpr}).to_csv(os.path.join(OUT,f"roc_{version}_{name}.csv"),index=False)
        pd.DataFrame({"predicted":cpred,"observed":ctrue}).to_csv(os.path.join(OUT,f"cal_{version}_{name}.csv"),index=False)
        calbins.to_csv(os.path.join(OUT,f"calibration_bins_{version}_{name}.csv"),index=False)
metrics_df=pd.DataFrame(metrics); metrics_df.to_csv(os.path.join(OUT,"model_metrics.csv"),index=False)
mb=metrics_df.pivot(index="model",columns="version",values=["roc_auc","average_precision","brier","ece_10"]).reset_index(); mb.columns=["_".join([str(x) for x in col if x]) for col in mb.columns]; mb.to_csv(os.path.join(OUT,"model_version_delta.csv"),index=False)

# SHAP explanations for GBM
if os.environ.get("SKIP_SHAP") != "1":
     for version in ["booking_time","near_arrival"]:
        pack=models[(version,"gradient_boosting")]; model=pack["model"]; pre=model.named_steps["pre"]; clf=model.named_steps["clf"]
        Xtest=pre.transform(df.loc[te,pack["features"]]); rng=np.random.default_rng(42); idx=rng.choice(Xtest.shape[0],size=min(1200,Xtest.shape[0]),replace=False); Xs=Xtest[idx]
        expl=shap.TreeExplainer(clf); sv=expl.shap_values(Xs)
        if isinstance(sv,list): sv=sv[1]
        if hasattr(sv,"toarray"): sv=sv.toarray()
        sv=np.asarray(sv)
        if sv.ndim==3: sv=np.squeeze(sv,axis=-1)
        fn=pre.get_feature_names_out(); imp=pd.DataFrame({"feature_encoded":fn,"mean_abs_shap":np.abs(sv).mean(axis=0),"mean_shap_logodds":sv.mean(axis=0)}).sort_values("mean_abs_shap",ascending=False)
        imp.to_csv(os.path.join(OUT,f"shap_encoded_{version}.csv"),index=False)
        def base_name(encoded):
            s=encoded.split("__",1)[-1]
            for pfx in ["country_","meal_","market_segment_","distribution_channel_","reserved_room_type_","deposit_type_","customer_type_","assigned_room_type_"]:
                if s.startswith(pfx): return pfx[:-1]
            return s
        imp["feature"]=imp.feature_encoded.map(base_name)
        grp=imp.groupby("feature",as_index=False).agg(mean_abs_shap=("mean_abs_shap","sum"),mean_shap_logodds=("mean_shap_logodds","sum")).sort_values("mean_abs_shap",ascending=False)
        grp.to_csv(os.path.join(OUT,f"shap_grouped_{version}.csv"),index=False)
        ptest=pack["p_test"]; test_indices=np.flatnonzero(te.to_numpy()); order=np.argsort(ptest); choices=[order[max(0,min(len(order)-1,int(q*len(order))))] for q in (.10,.50,.90)]
        Xc=pre.transform(df.iloc[test_indices[choices]][pack["features"]]); svc=expl.shap_values(Xc)
        if isinstance(svc,list): svc=svc[1]
        if hasattr(svc,"toarray"): svc=svc.toarray()
        svc=np.asarray(svc)
        if svc.ndim==3: svc=np.squeeze(svc,axis=-1)
        local=[]
        for j,ri in enumerate(test_indices[choices]):
            for k in np.argsort(np.abs(svc[j]))[::-1][:8]:
                enc=fn[k]; base=base_name(enc); val=df.iloc[ri][base] if base in df.columns else ""
                local.append({"case":j+1,"risk_percentile":[10,50,90][j],"booking_date":str(df.iloc[ri].booking_date.date()),"arrival_date":str(df.iloc[ri].arrival_date.date()),"actual_canceled":int(df.iloc[ri].is_canceled),"pred_cancel_prob":float(ptest[choices[j]]),"feature":base,"feature_encoded":enc,"feature_value":str(val),"shap_value_logodds":float(svc[j,k]),"direction":"increases cancellation risk" if svc[j,k]>0 else "decreases cancellation risk"})
        pd.DataFrame(local).to_csv(os.path.join(OUT,f"shap_local_{version}.csv"),index=False)


# Overbooking policy
scored=df.loc[te].copy().reset_index(drop=True)
pack=models[("booking_time","gradient_boosting")]
scored["p_cancel"]=pack["p_test"]; scored["p_show"]=1-scored.p_cancel
CAP_BASE=60.0
WALK_COSTS={"Low":125.0,"Base":200.0,"High":300.0}
completed=tr & df.is_canceled.eq(0) & df.adr.gt(0)
ROOM_RATE=float(df.loc[completed,"adr"].median())

def poisson_binomial_probs(ps):
    dist=np.array([1.0])
    for p in ps: dist=np.convolve(dist,[1-p,p])
    return dist

def expected_profit_table(cap,ks,walk_costs):
    rows=[]; groups=list(scored.groupby("arrival_date",sort=True))
    for k in ks:
        keep_count=int(cap+k); profit_by_cost={c:[] for c in walk_costs}; ex_show=[]; ex_walk=[]; acc=[]; realised_profit=[]; realised_walks=[]
        for _,g in groups:
            g2=g.sort_values("p_cancel").head(keep_count); ps=g2.p_show.to_numpy(); dist=poisson_binomial_probs(ps); s=np.arange(len(dist)); walk=np.maximum(s-cap,0); ex_show.append(float(np.dot(dist,s))); ex_walk.append(float(np.dot(dist,walk))); acc.append(len(g2))
            realised_show=float((1-g2.is_canceled).sum()); realised_walk=float(max(realised_show-cap,0)); realised_profit_base=ROOM_RATE*min(realised_show,cap)
            realised_profit.append(realised_profit_base); realised_walks.append(realised_walk)
            for c,wc in walk_costs.items(): profit_by_cost[c].append(float(np.dot(dist,ROOM_RATE*np.minimum(s,cap)-wc*walk)))
        for c,wc in walk_costs.items():
            rows.append({"capacity_rooms":cap,"overbooking_k":k,"walk_scenario":c,"walk_cost_eur":wc,"avg_nightly_profit_eur":np.mean(profit_by_cost[c]),"avg_expected_showups":np.mean(ex_show),"avg_expected_walked":np.mean(ex_walk),"avg_accepted_bookings":np.mean(acc),"realised_nightly_profit_eur":np.mean(realised_profit)-wc*np.mean(realised_walks),"realised_walks_per_night":np.mean(realised_walks)})
    return pd.DataFrame(rows)

ob=expected_profit_table(CAP_BASE,range(0,46),WALK_COSTS); ob.to_csv(os.path.join(OUT,"overbooking_profit_curve.csv"),index=False)
ob_best=ob.loc[ob.groupby("walk_scenario").avg_nightly_profit_eur.idxmax()].sort_values("walk_scenario").reset_index(drop=True); ob_best.to_csv(os.path.join(OUT,"overbooking_optima.csv"),index=False)
base_curve=ob[ob.walk_scenario=="Base"].sort_values("overbooking_k"); max_profit=float(base_curve.avg_nightly_profit_eur.max()); base0=float(base_curve.loc[base_curve.overbooking_k==0,"avg_nightly_profit_eur"].iloc[0]); basebest=base_curve.loc[base_curve.avg_nightly_profit_eur.idxmax()]; rec=base_curve[base_curve.avg_nightly_profit_eur>=.99*max_profit].iloc[0]
# Capacity/walk sensitivity
sens=[]; opt_sens=[]
for cap in [40,50,60,70,80]:
    tab=expected_profit_table(cap,range(0,46),WALK_COSTS)
    for _,r in tab.iterrows(): sens.append(r.to_dict())
    for sc in WALK_COSTS:
        rr=tab[tab.walk_scenario==sc].sort_values("avg_nightly_profit_eur").iloc[-1]
        opt_sens.append({"capacity_rooms":cap,"walk_scenario":sc,"walk_cost_eur":WALK_COSTS[sc],"optimal_k":int(rr.overbooking_k),"avg_nightly_profit_eur":float(rr.avg_nightly_profit_eur)})
pd.DataFrame(sens).to_csv(os.path.join(OUT,"overbooking_capacity_sensitivity.csv"),index=False); pd.DataFrame(opt_sens).to_csv(os.path.join(OUT,"overbooking_optimal_sensitivity.csv"),index=False)

# Deposit policy
scored["eligible_policy"]=(~scored.deposit_type.eq("Non Refund")) & (~scored.market_segment.isin(["Groups","Offline TA/TO"]))
scored["risk_tier"]=pd.cut(scored.p_cancel,bins=[-1,.20,.40,1.01],labels=["Low risk","Medium risk","High risk"])
scored["room_value"]=scored.adr.clip(lower=0)*scored.total_nights.clip(lower=0)
ASSUMPTIONS=[
    {"tier":"Low risk","risk_band":"<20%","deposit_pct":0.00,"cancel_deterrence":0.00,"booking_loss":0.00},
    {"tier":"Medium risk","risk_band":"20%-40%","deposit_pct":0.10,"cancel_deterrence":0.20,"booking_loss":0.04},
    {"tier":"High risk","risk_band":">40%","deposit_pct":0.25,"cancel_deterrence":0.35,"booking_loss":0.08},
]
REC_BASE=.50
rows=[]
for a in ASSUMPTIONS:
    m=scored.risk_tier.eq(a["tier"]) & scored.eligible_policy; n=int(m.sum()); p=scored.loc[m,"p_cancel"].to_numpy(); v=scored.loc[m,"room_value"].to_numpy(); exp_cancel=float(p.sum()); avgp=float(np.mean(p)) if n else np.nan; actual=float(scored.loc[m,"is_canceled"].mean()) if n else np.nan; avgval=float(np.mean(v)) if n else 0
    prevented=exp_cancel*(1-a["booking_loss"])*a["cancel_deterrence"]; lost=n*a["booking_loss"]; retained=prevented*avgval*REC_BASE; lostrev=lost*(1-avgp)*avgval; net=retained-lostrev
    rows.append({"tier":a["tier"],"risk_band":a["risk_band"],"eligible_bookings":n,"share_eligible":n/max(1,int(scored.eligible_policy.sum())),"avg_pred_cancel":avgp,"observed_cancel_rate":actual,"deposit_pct":a["deposit_pct"],"assumed_cancel_reduction":a["cancel_deterrence"],"assumed_booking_loss_rate":a["booking_loss"],"expected_cancellations":exp_cancel,"expected_cancellations_prevented":prevented,"expected_bookings_lost":lost,"avg_room_value_eur":avgval,"retained_revenue_eur":retained,"lost_revenue_eur":lostrev,"net_incremental_revenue_eur":net,"recommendation":"No deposit" if a["tier"]=="Low risk" else ("10% deposit" if a["tier"]=="Medium risk" else "25% deposit")})
dep=pd.DataFrame(rows); dep.to_csv(os.path.join(OUT,"deposit_policy_tiers.csv"),index=False)
base_total={"eligible_test_bookings":int(scored.eligible_policy.sum()),"expected_cancellations_prevented":float(dep.expected_cancellations_prevented.sum()),"expected_bookings_lost":float(dep.expected_bookings_lost.sum()),"net_incremental_revenue_eur":float(dep.net_incremental_revenue_eur.sum())}
elig=scored.eligible_policy.to_numpy(); baseline_rev=float((scored.loc[elig,"p_show"]*scored.loc[elig,"room_value"]).sum()); base_total["baseline_expected_room_revenue_eur"]=baseline_rev; base_total["policy_expected_room_revenue_eur"]=baseline_rev+base_total["net_incremental_revenue_eur"]; base_total["baseline_cancel_value_loss_eur"]=float((scored.loc[elig,"p_cancel"]*scored.loc[elig,"room_value"]).sum()); base_total["recovery_fraction_base"]=REC_BASE
pd.DataFrame([base_total]).to_csv(os.path.join(OUT,"deposit_policy_summary.csv"),index=False)

# Explicit low/base/high scenarios — assumptions are scenario parameters, not estimates.
scenario_defs=[
    ("Conservative",.10,.20,.06,.12,.30),
    ("Base",.20,.35,.04,.08,.50),
    ("Optimistic",.30,.50,.02,.04,.70),
]
srows=[]
for name,md,hd,ml,hl,recov in scenario_defs:
    prev=lost=net=0; nm=nh=0
    for _,r in dep.iterrows():
        if r.tier=="Low risk": continue
        det,loss=(md,ml) if r.tier=="Medium risk" else (hd,hl)
        pv=r.expected_cancellations*(1-loss)*det; ls=r.eligible_bookings*loss; tn=pv*r.avg_room_value_eur*recov-ls*(1-r.avg_pred_cancel)*r.avg_room_value_eur
        prev+=pv; lost+=ls; net+=tn
        if r.tier=="Medium risk": nm=tn
        else: nh=tn
    srows.append({"scenario":name,"medium_deterrence":md,"high_deterrence":hd,"medium_booking_loss":ml,"high_booking_loss":hl,"revenue_recovery_fraction":recov,"cancellations_prevented":prev,"bookings_lost":lost,"net_medium_tier_eur":nm,"net_high_tier_eur":nh,"net_incremental_revenue_eur":net})
sensdep=pd.DataFrame(srows); sensdep.to_csv(os.path.join(OUT,"deposit_sensitivity.csv"),index=False)
# Break-even recovery fraction for base response rates.
base_no_recovery=0; base_revenue_coeff=0
for _,r in dep.iterrows():
    if r.tier=="Low risk": continue
    # net = recovery*(prevented*avgval) - lostrev
    base_revenue_coeff += r.expected_cancellations*(1-r.assumed_booking_loss_rate)*r.assumed_cancel_reduction*r.avg_room_value_eur
    base_no_recovery += r.lost_revenue_eur
break_even=base_no_recovery/base_revenue_coeff if base_revenue_coeff else np.nan
pd.DataFrame([{"break_even_revenue_recovery_fraction_base_response":break_even}]).to_csv(os.path.join(OUT,"deposit_break_even.csv"),index=False)

# Trade-off table for recommended scopes.
trade=[]
for scope,tiers in [("Medium risk only",["Medium risk"]),("Medium + High risk",["Medium risk","High risk"])]:
    chosen=dep[dep.tier.isin(tiers)]; trade.append({"scope":scope,"expected_cancellations_prevented":float(chosen.expected_cancellations_prevented.sum()),"expected_bookings_lost":float(chosen.expected_bookings_lost.sum()),"net_incremental_revenue_eur":float(chosen.net_incremental_revenue_eur.sum())})
pd.DataFrame(trade).to_csv(os.path.join(OUT,"deposit_tradeoff.csv"),index=False)

# Fairness ablation without country
feat_no_country=[f for f in base_features if f!="country"]
pre=preprocessor(feat_no_country); gb=Pipeline([("pre",pre),("clf",LGBMClassifier(n_estimators=350,learning_rate=.035,num_leaves=31,subsample=.9,colsample_bytree=.85,reg_lambda=2.0,random_state=42,n_jobs=-1,verbosity=-1))]); gb.fit(df.loc[tr,feat_no_country],y[tr]); iso=IsotonicRegression(out_of_bounds="clip").fit(gb.predict_proba(df.loc[ca,feat_no_country])[:,1],y[ca]); p=iso.predict(gb.predict_proba(df.loc[te,feat_no_country])[:,1]); a=df.loc[te,["country","is_canceled"]].copy(); a["p_cancel"]=p; a["country_group"]=np.where(a.country.isin(["PRT","GBR","FRA","DEU","ESP"]),a.country,"Other"); a["high_risk"]=(a.p_cancel>.40).astype(int)
abl=a.groupby("country_group").agg(n=("p_cancel","size"),mean_predicted=("p_cancel","mean"),observed=("is_canceled","mean"),share_high_risk=("high_risk","mean")).reset_index(); abl.loc[len(abl)]=["ALL",len(a),a.p_cancel.mean(),a.is_canceled.mean(),a.high_risk.mean()]; abl["auc_all_rows"]=np.nan; abl.loc[abl.country_group=="ALL","auc_all_rows"]=roc_auc_score(a.is_canceled,a.p_cancel); abl.to_csv(os.path.join(OUT,"ablation_no_country.csv"),index=False)

# Fairness audit with country in model
fair=scored[["country","is_canceled","p_cancel","eligible_policy","risk_tier"]].copy(); fair["high_risk"]=(fair.risk_tier=="High risk").astype(int); fair["country_group"]=np.where(fair.country.isin(["PRT","GBR","FRA","DEU","ESP"]),fair.country,"Other"); fair["high_risk"]=(fair.risk_tier=="High risk").astype(int); fair["deposit_pct"]=fair.risk_tier.astype(str).map({"Low risk":0.0,"Medium risk":.10,"High risk":.25}).where(fair.eligible_policy,0.0).astype(float)
def summarise(g):
    auc=roc_auc_score(g.is_canceled,g.p_cancel) if g.is_canceled.nunique()>1 and len(g)>=100 else np.nan
    return pd.Series({"n":len(g),"mean_predicted":g.p_cancel.mean(),"observed_cancel":g.is_canceled.mean(),"calibration_gap":g.p_cancel.mean()-g.is_canceled.mean(),"auc":auc,"share_high_risk":g.high_risk.mean(),"avg_deposit_pct":g.deposit_pct.mean(),"share_policy_eligible":g.eligible_policy.mean()})
fr=fair.groupby("country_group").apply(summarise,include_groups=False).reset_index(); overall=summarise(fair); overall["country_group"]="ALL"; fr=pd.concat([fr,overall.to_frame().T],ignore_index=True); fr.to_csv(os.path.join(OUT,"fairness_by_country.csv"),index=False)

# Persist scored test data with country for the fairness audit.
score_cols=["country","booking_date","arrival_date","lead_time","market_segment","distribution_channel","deposit_type","customer_type","adr","total_nights","is_canceled","p_cancel","p_show","eligible_policy","risk_tier","room_value"]
scored[score_cols].to_csv(os.path.join(OUT,"test_scored_bookings.csv"),index=False)

# Charts for report/dashboard.
ob_base=ob[ob.walk_scenario=="Base"]
plt.figure(figsize=(8,5));
for sc in ["Low","Base","High"]:
    g=ob[ob.walk_scenario==sc]; plt.plot(g.overbooking_k,g.avg_nightly_profit_eur,marker="o",label=sc)
plt.axvline(int(rec.overbooking_k),linestyle="--",linewidth=1,label=f"Recommended k={int(rec.overbooking_k)}")
plt.xlabel("Overbooking level k"); plt.ylabel("Expected nightly room profit (€)"); plt.title("Expected overbooking profit — assumed 60-room capacity"); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT,"profit_curve.png"),dpi=170); plt.close()
plt.figure(figsize=(8,5));
for name,label in [("logistic","Logistic regression"),("gradient_boosting","Gradient boosting")]:
    g=pd.read_csv(os.path.join(OUT,f"pr_booking_time_{name}.csv")); plt.plot(g.recall,g.precision,label=label)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall — booking-time models"); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT,"pr_curve_booking_time.png"),dpi=170); plt.close()
plt.figure(figsize=(8,5));
for name,label in [("logistic","Logistic regression"),("gradient_boosting","Gradient boosting")]:
    g=pd.read_csv(os.path.join(OUT,f"cal_booking_time_{name}.csv")); plt.plot(g.predicted,g.observed,marker="o",label=label)
plt.plot([0,1],[0,1],linestyle="--",label="Perfect calibration"); plt.xlabel("Predicted cancellation probability"); plt.ylabel("Observed cancellation rate"); plt.title("Calibration — booking-time models"); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(OUT,"calibration_booking_time.png"),dpi=170); plt.close()
plt.figure(figsize=(7,5)); plt.scatter(sensdep.cancellations_prevented,sensdep.bookings_lost,s=90)
for _,r in sensdep.iterrows(): plt.annotate(r.scenario,(r.cancellations_prevented,r.bookings_lost),xytext=(7,7),textcoords="offset points",fontsize=9)
plt.xlabel("Expected cancellations prevented"); plt.ylabel("Expected bookings lost"); plt.title("Deposit policy scenario trade-off"); plt.tight_layout(); plt.savefig(os.path.join(OUT,"deposit_tradeoff.png"),dpi=170); plt.close()

meta={
 "clean_file":clean_file,
 "features_booking_time":base_features,
 "features_near_arrival":near_features,
 "split":{"train_end":str(cut70.date()),"calibration_end":str(cut85.date()),"test_start":str((cut85+pd.Timedelta(days=1)).date()),"train_n":int(tr.sum()),"calibration_n":int(ca.sum()),"test_n":int(te.sum())},
 "leakage_correction":["required_car_parking_spaces","total_of_special_requests"],
 "overbooking":{"base_capacity_rooms":CAP_BASE,"room_rate_eur":ROOM_RATE,"walk_costs_eur":WALK_COSTS,"base_optimal_k":int(basebest.overbooking_k),"recommended_k_within_1pct":int(rec.overbooking_k),"base_optimal_profit_eur":float(basebest.avg_nightly_profit_eur),"recommended_profit_eur":float(rec.avg_nightly_profit_eur),"base_no_overbook_profit_eur":base0,"optimal_profit_uplift_eur_per_night":float(basebest.avg_nightly_profit_eur-base0),"recommended_profit_uplift_eur_per_night":float(rec.avg_nightly_profit_eur-base0),"optimal_realised_walks_per_night":float(basebest.realised_walks_per_night),"recommended_realised_walks_per_night":float(rec.realised_walks_per_night)},
 "deposit_policy":{"base_assumptions":ASSUMPTIONS,"base_recovery_fraction":REC_BASE,"scenario_names":[x[0] for x in scenario_defs],"break_even_recovery_fraction":break_even,"policy_exclusions":["Non Refund deposit type","Groups market segment","Offline TA/TO market segment"]},
 "external_data":{"weather":"not supplied; no values fabricated","google_trends":"not supplied; loader supports last completed week join"},
 "canonical_script":"src/03_model_policy_analysis.py"
}
json.dump(meta,open(os.path.join(OUT,"run_metadata.json"),"w"),indent=2)
print(metrics_df.to_string(index=False)); print("OVERBOOKING\n",ob_best.to_string(index=False)); print("RECOMMENDED K",int(rec.overbooking_k)); print("DEPOSIT SCENARIOS\n",sensdep.to_string(index=False)); print("BREAK EVEN RECOVERY",break_even)
