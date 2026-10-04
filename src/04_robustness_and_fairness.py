import os,pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..")); OUT=os.path.join(ROOT,"outputs")
main=pd.read_csv(os.path.join(OUT,"model_metrics.csv")); main.insert(0,"dataset","Duplicates removed (main)")
nd=pd.read_csv(os.path.join(OUT+"_nodedup","model_metrics.csv")); nd.insert(0,"dataset","Duplicates kept")
pd.concat([main,nd],ignore_index=True).to_csv(os.path.join(OUT,"robustness_dedup.csv"),index=False)
s=pd.read_csv(os.path.join(OUT,"test_scored_bookings.csv")); top=["PRT","GBR","FRA","DEU","ESP"]; s["country_group"]=np.where(s.country.isin(top),s.country,"Other"); s["high_risk"]=s.risk_tier.astype(str).eq("High risk").astype(int); s["deposit_pct"]=s.risk_tier.astype(str).map({"Low risk":0.0,"Medium risk":.10,"High risk":.25}).where(s.eligible_policy,0.0).astype(float)
def summary(g):
    auc=roc_auc_score(g.is_canceled,g.p_cancel) if len(g)>=100 and g.is_canceled.nunique()>1 else np.nan
    return pd.Series({"n":len(g),"mean_predicted":g.p_cancel.mean(),"observed_cancel":g.is_canceled.mean(),"calibration_gap":g.p_cancel.mean()-g.is_canceled.mean(),"auc":auc,"share_high_risk":g.high_risk.mean(),"avg_deposit_pct":g.deposit_pct.mean(),"share_policy_eligible":g.eligible_policy.mean()})
f=s.groupby("country_group").apply(summary,include_groups=False).reset_index(); a=summary(s); a["country_group"]="ALL"; fair=pd.concat([f,a.to_frame().T],ignore_index=True); fair.to_csv(os.path.join(OUT,"fairness_by_country.csv"),index=False)
print('ROBUSTNESS'); print(pd.concat([main,nd]).to_string(index=False)); print('\nFAIRNESS'); print(fair.round(3).to_string(index=False))
