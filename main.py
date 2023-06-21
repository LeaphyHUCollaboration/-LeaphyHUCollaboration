# https://www.datacourses.com/multiple-regression-in-statsmodels-4158/
import numpy as np
import pandas as pd
# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

file = 'real-data.csv'
# learning performance
df = pd.read_csv(file, usecols=["age", "motivation", "interest", "block programming", "code programming", "self confidence", "learning performance"])
df.isnull().sum()*100/df.shape[0]


plt.clf()
sns.heatmap(df.corr(), cmap="YlGnBu", annot = True)
title = "graphs/" + "heatmap" + '.png'
plt.savefig(title, bbox_inches='tight')
