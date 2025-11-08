import pandas as pd 
import os 
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

os.chdir("/Users/paulbenjamins/Desktop/data_afm")
df = pd.read_csv("factors.csv")
df = df.drop(columns=(["Unnamed: 0","HML_Europe","HML_Global","HML_US","SMB_Europe","SMB_Global","SMB_US","UMD_Europe","UMD_Global","UMD_US","RF"]))


#a
res = df.loc[:,"Mkt_Europe":].describe().iloc[np.r_[1:3,4:7],:]
res.loc["IQR"] = res.iloc[4] - res.iloc[2]
res=res.rename(index={"50%":"median"})
res = res.iloc[[3,5,0,1],:]

#b) density hist!!!!!!!
def hist (x,title):
    os.chdir("/Users/paulbenjamins/Desktop/data_afm/png")
    plt.figure(figsize=(8, 5))
    plt.style.use('seaborn-v0_8-whitegrid')  
    plt.hist(x, bins="auto",density=True, color='blue', edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('excess return', fontsize=12)
    plt.ylabel('probability density (%)', fontsize=12)
    plt.gca().yaxis.get_major_formatter().set_useOffset(False)   
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.savefig(title,dpi=600)
    plt.show()
    
    
    
hist(df["BAB_US"],"BAB US")
hist(df["Mkt_US"],"Market US")

hist(df["BAB_Europe"],"BAB Europe")
hist(df["Mkt_Europe"],"Market Europe")

hist(df["Mkt_Global"],"Market Global")
hist(df["BAB_Global"],"BAB Global")


#d,e) linear regression and coefficients

def linear_regression_plot (a,b,title,store):
    
    os.chdir("/Users/paulbenjamins/Desktop/data_afm/png")
    
    X = sm.add_constant(a)
    model = sm.OLS(b, X).fit()

    Y = model.predict(X)

    model.summary()

    plt.scatter(a, b, alpha=0.8, s = 7)
    plt.plot(a, Y, color="red")
    
    
    plt.axhline(0, linewidth=1, color="black")
    plt.axvline(0, linewidth=1, color="black")
    plt.xlabel("market excess return")
    plt.ylabel("BAB excess return")
    plt.title(rf"{title} $\alpha$ = {model.params[0]:.4f}  $\beta$ = {model.params[1]:.4f}")
    
    plt.tight_layout()
    
    plt.savefig(store,dpi=600)
    
    #plt.show()
    
    
linear_regression_plot (df["Mkt_Global"], df["BAB_Global"],"GLOBAL","linear regression GLOBAL")
linear_regression_plot (df["Mkt_Europe"], df["BAB_Europe"],"EUROPE","linear regression EUROPE")
linear_regression_plot (df["Mkt_US"], df["BAB_US"],"US","linear regression US")



#f)
df["D.Recession"].describe().to_frame()

df["P.Recession"].describe().to_frame()








