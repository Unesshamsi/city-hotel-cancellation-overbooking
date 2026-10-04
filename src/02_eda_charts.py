"""02_eda_charts.py — eight charts and managerial insights for the cleaned City Hotel data."""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT_CHARTS = os.path.join(ROOT, "charts")
OUT_DATA = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_CHARTS, exist_ok=True)
os.makedirs(OUT_DATA, exist_ok=True)

df = pd.read_csv(os.path.join(OUT_DATA, "city_hotel_clean.csv"), parse_dates=["arrival_date", "booking_date"])
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#6b6a66", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
insights=[]
def savefig(path): plt.tight_layout(); plt.savefig(path,dpi=160); plt.close()
def note(n,t): insights.append(f"**Chart {n}.** {t}")

# 1 lead time
bins=[0,7,30,90,180,365,1000]; labels=["0–7d","8–30d","31–90d","91–180d","181–365d","365d+"]
df["lead_bucket"]=pd.cut(df.lead_time,bins=bins,labels=labels,include_lowest=True)
g=df.groupby("lead_bucket",observed=True).is_canceled.mean()
plt.figure(figsize=(7,4.2)); plt.bar(g.index.astype(str),g.values,color=BLUE,width=.6)
for i,v in enumerate(g.values): plt.text(i,v+.012,f"{v:.0%}",ha="center",fontsize=9,color=MUTED)
plt.ylabel("Cancellation rate"); plt.title("Cancellation rate rises sharply with lead time"); savefig(os.path.join(OUT_CHARTS,"01_cancel_by_lead_time.png"))
note(1,f"Bookings made 365+ days ahead cancel {g.iloc[-1]:.0%} of the time versus {g.iloc[0]:.0%} within a week; lead time is therefore an early operational risk signal.")

# 2 deposit type
g2=df.groupby("deposit_type").is_canceled.agg(["mean","size"])
plt.figure(figsize=(6,4.5)); cols=[BLUE,RED,ORANGE][:len(g2)]
plt.bar(g2.index,g2["mean"],color=cols,width=.55)
for i,(m,s) in enumerate(zip(g2["mean"],g2["size"])): plt.text(i,m+.03,f"{m:.0%}\n(n={s:,})",ha="center",fontsize=9,color=MUTED)
plt.ylabel("Cancellation rate"); plt.ylim(0,1.15); plt.title("Non-Refund cancellations are a process signal"); savefig(os.path.join(OUT_CHARTS,"02_cancel_by_deposit_type.png"))
note(2,f"Non-Refund bookings cancel {g2.loc['Non Refund','mean']:.0%} historically, but this is concentrated in structured/allotment channels; it should not be treated as causal evidence that a consumer deposit fails.")

# 3 segment
g3=df.groupby("market_segment").is_canceled.agg(["mean","size"]); g3=g3[g3["size"]>=50].sort_values("mean")
plt.figure(figsize=(7.5,4.5)); plt.barh(g3.index,g3["mean"],color=AQUA)
for i,(m,s) in enumerate(zip(g3["mean"],g3["size"])): plt.text(m+.01,i,f"{m:.0%} (n={s:,})",va="center",fontsize=9,color=MUTED)
plt.xlabel("Cancellation rate"); plt.title("Cancellation risk varies materially by channel"); savefig(os.path.join(OUT_CHARTS,"03_cancel_by_segment.png"))
note(3,"Segment differences are operationally useful but can proxy channel, contract and allotment structure; treatment decisions should not assume a segment effect is purely guest behaviour.")

# 4 monthly trend — corrected wording
df["arrival_ym"]=df.arrival_date.dt.to_period("M").dt.to_timestamp(); monthly=df.groupby("arrival_ym").agg(cancel_rate=("is_canceled","mean"),holiday_arrivals=("is_arrival_holiday","sum")); yearly=df.groupby(df.arrival_date.dt.year).is_canceled.mean()
fig,ax1=plt.subplots(figsize=(9,4.5)); ax1.plot(monthly.index,monthly.cancel_rate,color=BLUE,linewidth=2,marker="o",markersize=3); ax1.set_ylabel("Cancellation rate",color=BLUE); ax1.set_title("Monthly cancellation rate varies over the historical window")
ax2=ax1.twinx(); ax2.bar(monthly.index,monthly.holiday_arrivals,color=ORANGE,alpha=.3,width=20); ax2.set_ylabel("Holiday arrivals",color=ORANGE); ax2.grid(False); savefig(os.path.join(OUT_CHARTS,"04_monthly_trend_holidays.png"))
note(4,f"The 2017 Jan–Aug period has a higher cancellation rate ({yearly.loc[2017]:.0%}) than 2016 ({yearly.loc[2016]:.0%}), while 2015 is only Jul–Dec ({yearly.loc[2015]:.0%}); the partial-year coverage means this is a descriptive historical signal, not proof of a monotonic trend.")

# 5 ADR
plt.figure(figsize=(6.5,4.5)); data=[df.loc[df.is_canceled==0,"adr"].clip(upper=400),df.loc[df.is_canceled==1,"adr"].clip(upper=400)]; bp=plt.boxplot(data,tick_labels=["Completed","Canceled"],patch_artist=True,widths=.5,medianprops=dict(color=INK))
for patch,c in zip(bp["boxes"],[BLUE,RED]): patch.set_facecolor(c); patch.set_alpha(.55)
plt.ylabel("ADR (€; display clipped at 400)"); plt.title("Cancelled bookings carry somewhat higher ADR"); savefig(os.path.join(OUT_CHARTS,"05_adr_by_outcome.png"))
med_c=df.loc[df.is_canceled==0,"adr"].median(); med_x=df.loc[df.is_canceled==1,"adr"].median(); note(5,f"Median ADR is €{med_x:.0f} for cancelled versus €{med_c:.0f} for completed bookings; higher-value reservations deserve attention when room value at risk is considered.")

# 6 non-refund lead time
plt.figure(figsize=(6.5,4.5)); nr=df.loc[df.deposit_type=="Non Refund","lead_time"].clip(upper=500); nd=df.loc[df.deposit_type=="No Deposit","lead_time"].clip(upper=500)
plt.hist(nd,bins=40,alpha=.55,label=f"No Deposit (median {nd.median():.0f}d)",color=BLUE,density=True); plt.hist(nr,bins=40,alpha=.55,label=f"Non Refund (median {nr.median():.0f}d)",color=RED,density=True); plt.xlabel("Lead time (days, clipped at 500)"); plt.ylabel("Density"); plt.legend(frameon=False); plt.title("Non-Refund bookings are booked far earlier"); savefig(os.path.join(OUT_CHARTS,"06_leadtime_nonrefund_vs_nodeposit.png"))
note(6,f"Non-Refund bookings have a median lead time of {nr.median():.0f} days versus {nd.median():.0f} for No Deposit, consistent with structured advance allotments rather than a clean consumer-policy comparison.")

# 7 composition
nr_df=df[df.deposit_type=="Non Refund"]; country_share=nr_df.country.value_counts(normalize=True).head(6); seg_share=nr_df.market_segment.value_counts(normalize=True)
fig,axes=plt.subplots(1,2,figsize=(10,4.5)); axes[0].bar(country_share.index,country_share.values,color=VIOLET); axes[0].set_title("Non-Refund: country mix"); axes[0].set_ylabel("Share")
for i,v in enumerate(country_share.values): axes[0].text(i,v+.01,f"{v:.0%}",ha="center",fontsize=8,color=MUTED)
axes[1].bar(seg_share.index,seg_share.values,color=MAGENTA); axes[1].set_title("Non-Refund: segment mix"); axes[1].tick_params(axis="x",rotation=30)
for i,v in enumerate(seg_share.values): axes[1].text(i,v+.01,f"{v:.0%}",ha="center",fontsize=8,color=MUTED)
savefig(os.path.join(OUT_CHARTS,"07_nonrefund_composition.png")); nr_pct=(seg_share.get("Groups",0)+seg_share.get("Offline TA/TO",0)); note(7,f"{country_share.iloc[0]:.0%} of Non-Refund bookings are from {country_share.index[0]} and {nr_pct:.0%} are Groups/Offline TA-TO; this supports treating the category as a process/channel signal rather than a consumer experiment.")

# 8 holiday proximity
hbins=[0,3,7,14,30,1000]; hlabels=["0–3d","4–7d","8–14d","15–30d","30d+"]; df["holiday_bucket"]=pd.cut(df.days_to_nearest_holiday,bins=hbins,labels=hlabels,include_lowest=True); g8=df.groupby("holiday_bucket",observed=True).is_canceled.mean()
plt.figure(figsize=(7,4.2)); plt.bar(g8.index.astype(str),g8.values,color=GREEN,width=.6)
for i,v in enumerate(g8.values): plt.text(i,v+.012,f"{v:.0%}",ha="center",fontsize=9,color=MUTED)
plt.ylabel("Cancellation rate"); plt.title("Cancellation differs around public-holiday dates"); savefig(os.path.join(OUT_CHARTS,"08_cancel_by_holiday_proximity.png"))
note(8,f"Bookings arriving within 3 days of a holiday cancel {g8.iloc[0]:.0%} versus {g8.iloc[-1]:.0%} for 30+ days away; this is useful as planning context, not a causal holiday effect.")

with open(os.path.join(OUT_DATA,"eda_insights.md"),"w",encoding="utf-8") as f: f.write("# EDA — 8 charts and managerial insights\n\n"+"\n\n".join(insights)+"\n")
print("Saved 8 charts and EDA insights")
