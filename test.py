import statsmodels
import statsmodels.formula.api as smf

import pandas as pd
file = 'real-data.csv'
df = pd.read_csv(file, usecols=["age", "motivation", "interest", "block programming", "code programming", "self confidence", "learning performance"])
df.isnull().sum()*100/df.shape[0]
#create DataFrame
# df = pd.DataFrame({'hours': [1, 2, 2, 4, 2, 1, 5, 4, 2, 4, 4, 3, 6],
#                    'exams': [1, 3, 3, 5, 2, 2, 1, 1, 0, 3, 4, 3, 2],
#                    'score': [76, 78, 85, 88, 72, 69, 94, 94, 88, 92, 90, 75, 96]})

#view head of DataFrame
df.head()


#define predictor and response variables
res = smf.ols(formula= "'learning performance' ~ age + motivation + interest + 'block programming' + 'code programming' + 'self confidence'", data=df).fit()
print(res.summary())#view model summary
