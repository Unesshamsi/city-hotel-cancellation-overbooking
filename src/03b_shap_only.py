"""Small, deterministic SHAP pass for the canonical models. Uses the same split/features/model settings as 03_model_policy_analysis.py but 400 sampled test rows for speed."""
import os, numpy as np, pandas as pd, shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..")); OUT=os.path.join(ROOT,"outputs")
df=pd.read_csv(os.path.join(ROOT,"data","processed","city_hotel_clean.csv"),parse_dates=["arrival_date","booking_date"]); y=df.is_canceled.astype(int)
base=["lead_time","arrival_month","arrival_dow","stays_in_weekend_nights","stays_in_week_nights","total_nights","adults","children","babies","total_guests","meal","country","market_segment","distribution_channel","is_repeated_guest","previous_cancellations","previous_bookings_not_canceled","reserved_room_type","deposit_type","has_agent","has_company","days_in_waiting_list","customer_type","adr","is_free_stay","is_zero_night","booking_dow","booking_month","is_arrival_holiday","days_to_nearest_holiday"]
near=base+["assigned_room_type","booking_changes","room_type_changed","required_car_parking_spaces","total_of_special_requests"]
c70=df.booking_date.quantile(.70); c85=df.booking_date.quantile(.85); tr=df.booking_date<=c70; te=df.booking_date>c85
for version,features in [("booking_time",base),("near_arrival",near)]:
    cats=[c for c in features if df[c].dtype=="object"]; nums=[c for c in features if c not in cats]
    pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),nums),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",min_frequency=3))]),cats)])
    gb=Pipeline([("pre",pre),("clf",LGBMClassifier(n_estimators=350,learning_rate=.035,num_leaves=31,subsample=.9,colsample_bytree=.85,reg_lambda=2,random_state=42,n_jobs=-1,verbosity=-1))]); gb.fit(df.loc[tr,features],y[tr])
    pp=gb.named_steps["pre"]; clf=gb.named_steps["clf"]; Xt=pp.transform(df.loc[te,features]); rng=np.random.default_rng(42); idx=rng.choice(Xt.shape[0],size=min(400,Xt.shape[0]),replace=False); sv=shap.TreeExplainer(clf).shap_values(Xt[idx]);
    if isinstance(sv,list): sv=sv[1]
    if hasattr(sv,'toarray'): sv=sv.toarray()
    sv=np.asarray(sv); sv=np.squeeze(sv,-1) if sv.ndim==3 else sv; fn=pp.get_feature_names_out()
    imp=pd.DataFrame({"feature_encoded":fn,"mean_abs_shap":np.abs(sv).mean(axis=0),"mean_shap_logodds":sv.mean(axis=0)})
    def base_name(encoded):
        s=encoded.split("__",1)[-1]
        for pfx in ["country_","meal_","market_segment_","distribution_channel_","reserved_room_type_","deposit_type_","customer_type_","assigned_room_type_"]:
            if s.startswith(pfx): return pfx[:-1]
        return s
    imp["feature"]=imp.feature_encoded.map(base_name); imp.sort_values("mean_abs_shap",ascending=False).to_csv(os.path.join(OUT,f"shap_encoded_{version}.csv"),index=False); imp.groupby("feature",as_index=False).agg(mean_abs_shap=("mean_abs_shap","sum"),mean_shap_logodds=("mean_shap_logodds","sum")).sort_values("mean_abs_shap",ascending=False).to_csv(os.path.join(OUT,f"shap_grouped_{version}.csv"),index=False)
print('SHAP done')
