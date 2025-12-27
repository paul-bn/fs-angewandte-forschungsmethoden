#paulbenjamins
import pandas as pd 
import os 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_breuschpagan
import quantstats as qs
from scipy import stats
from scipy.stats import wilcoxon 
from statistics import geometric_mean
from scipy.stats import kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor

#lokaler eigener Pfad
os.chdir("/Users/paulbenjamins/Desktop/data_afm")

#Rohdaten in DataFrame einlesen und Rohdatei anpassen
df = pd.read_csv("factors.csv")
df = df.drop(columns=(["Unnamed: 0","UMD_Europe","UMD_Global","UMD_US","D.Recession","P.Recession"]))
df["RF"] = df["RF"] / 100


###


#einfaches Regressionsmodel
X = sm.add_constant(df["Mkt_Europe"])   
model = sm.OLS(df["BAB_Europe"],X).fit()
Y = model.predict(X)

model.summary()


#Pearson Korrelationskoeffizeint
rho = df["Mkt_Europe"].corr(df["BAB_Europe"], method="pearson")


#Regressionsparamter für Excel-Export
results_df = pd.DataFrame({
    "coef": model.params,
    "p_value": model.pvalues,
    "r^2":model.rsquared,
    "r^2 adjusted": model.rsquared_adj,
    "pearson":rho,
    "n":model.nobs
    })
results_df.to_excel("einfache reg.xlsx")


#bp test, einfaches Regressionsmodell
het_breuschpagan(model.resid, model.model.exog)
    

#scatter plot und lineare Regressionsgerade
plt.scatter(df["Mkt_Europe"],df["BAB_Europe"] , alpha=0.8, s = 7)
plt.plot(df["Mkt_Europe"], Y, color="red")
    
plt.axhline(0, linewidth=1, color="black")
plt.axvline(0, linewidth=1, color="black")
plt.xlabel("Überschussrenditen des Marktportfolios")
plt.ylabel("Überschussrenditen der BAB-Strategie")
plt.tight_layout()
plt.figure(dpi=1000)
plt.show()


#qq plots
sm.qqplot(df["BAB_Europe"], line='45',fit=True)
plt.grid(True)
plt.title("BAB-Strategie")
plt.xlabel("Theoretische Quantile der Normalverteilung")
plt.ylabel("Empirische Quantile des Datensamples")
plt.show() 
    
sm.qqplot(df["Mkt_Europe"], line='45',fit=True)
plt.grid(True)
plt.title("Marktportfolio")
plt.xlabel("Theoretische Quantile der Normalverteilung")
plt.ylabel("Empirische Quantile des Datensamples")
plt.show() 


#Kurtosis nach Fisher
kurtosis(df["BAB_Europe"], fisher=True)
kurtosis(df["Mkt_Europe"], fisher=True)


#Schiefe
df["BAB_Europe"].skew()
df["Mkt_Europe"].skew()


#Hypothesentests für Excel-Export
tests_final=pd.DataFrame({
    "adf": [
        adfuller(df["BAB_Europe"])[1],
        adfuller(df["Mkt_Europe"])[1],
        ],
    "ttest": [
        stats.ttest_1samp(df["BAB_Europe"], popmean=0,alternative="greater")[1],
        stats.ttest_1samp(df["Mkt_Europe"], popmean=0,alternative="greater")[1],
        ],
    "wilcoxon":[
        wilcoxon(df["BAB_Europe"],alternative="greater")[1],
        wilcoxon(df["Mkt_Europe"],alternative="greater")[1],
        ],
    "shapiro":[
        shapiro(df["BAB_Europe"])[1],
        shapiro(df["Mkt_Europe"])[1],
        ]},
    index = ["BAB","Market"]
    )
tests_final.transpose().to_excel("tests.xlsx")


#Hypothesentest der Renditedifferenz
wilcoxon(df["BAB_Europe"]-df["Mkt_Europe"],alternative="greater")[1]
stats.ttest_1samp(df["BAB_Europe"]-df["Mkt_Europe"], popmean=0,alternative="greater")[1]


#Deskriptive Statistik für Excel-Export, (Annualisierung erfolgte in Excel)
res = df[["Mkt_Europe","BAB_Europe"]].describe().iloc[np.r_[1:3,4:7],:]
res.loc["IQR"] = res.iloc[4] - res.iloc[2]
res=res.rename(index={"50%":"median"})
res = res.iloc[[3,5,0,1],:]

for col in res.columns:
    res.loc["geometric_mean",col] = geometric_mean(1 + df[col]) - 1

res[["BAB_Europe","Mkt_Europe"]]*100

res.to_excel("res.xlsx")


#boxplots
fig, ax = plt.subplots(figsize=(10, 10))
plt.style.use("seaborn-v0_8-whitegrid")

ax.boxplot(
    [df["BAB_Europe"], df["Mkt_Europe"]],
    labels=["BAB-Strategie", "Marktportfolio"],
    widths=0.3,
    patch_artist=False,        
    boxprops=dict(color="black", linewidth=1.5),
    medianprops=dict(color="red", linewidth=2),  
    whiskerprops=dict(color="black", linewidth=1.2),
    capprops=dict(color="black", linewidth=1.2),
    flierprops=dict(marker='o', markersize=4, markerfacecolor='black', markeredgecolor='black')
)

ax.tick_params(axis='both', labelsize=20)

ax.axhline(0, color='black', linewidth=1)
plt.show()


#Annualisierte Sharpe Ratio
qs.extend_pandas()
df["BAB_Europe"].sharpe(periods=12)
df["Mkt_Europe"].sharpe(periods=12)


#Multiples Regressionsmodel (Fama French 3-Faktor Modell)
A = sm.add_constant(df[["SMB_Europe","HML_Europe","Mkt_Europe"]])
ff_model = sm.OLS(df["BAB_Europe"],A).fit()
ff_model.summary()


#Regressionsparamter des multiplen Regressionsmodells für Excel-Export
R = np.sqrt(ff_model.rsquared)
results_ff_df = pd.DataFrame({
    "coef": ff_model.params,
    "p_value": ff_model.pvalues,
    "r^2": ff_model.rsquared,
    "r^2 adj.":ff_model.rsquared_adj, 
    "Wurzel aus Bestimmheitsmaß": R,
    "n": ff_model.nobs
})
results_ff_df.to_excel("multiple reg.xlsx")


#bp Test multiples Regressionsmodell
het_breuschpagan(ff_model.resid, ff_model.model.exog)


#vif Test, Multikollinearität der erklärenden Variablen im FF3
X = df[["Mkt_Europe","SMB_Europe","HML_Europe"]]
vif = pd.Series(
    [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    index=X.columns)
print(vif)


#Equity Curve
final = pd.DataFrame()

final["BAB_total_return"] = df["BAB_Europe"] + df["RF"]
final["Mkt_total_return"] = df["Mkt_Europe"] + df["RF"]
    
final["BAB"] = (1+final["BAB_total_return"]).cumprod().to_frame()
final["Mkt"] = (1+final["Mkt_total_return"]).cumprod().to_frame()

final["time"] = df["mdate"]
final = final.set_index("time")

plt.plot(final.index.astype(str), final[["BAB","Mkt"]].values, label=["BAB","Market"])
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(16))
plt.xticks(rotation=60)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()