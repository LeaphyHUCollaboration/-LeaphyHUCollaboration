import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm

from sklearn.feature_selection import f_regression
######################################## data preparation #########################################

# file = 'https://aegis4048.github.io/downloads/notebooks/sample_data/unconv_mv_v5.csv'

singular_items = ['age','motivation','interest','block_programming','code_programming','self_confidence']
combined_items = [['motivation', 'interest'], ['block programming','code programming']]

file = 'real-data.csv'
df = pd.read_csv(file, usecols=["age", "motivation", "interest", "block_programming", "code_programming", "self_confidence", "total"])
for i in singular_items:
    x = df[i].values.reshape(-1,1)
    y = df['total'].values

    ################################################ train #############################################

    ols = linear_model.LinearRegression()
    model = ols.fit(x, y)
    response = model.predict(x)

    ############################################## evaluate ############################################

    r2 = model.score(x, y)

    ############################################## plot ################################################

    plt.style.use('default')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(x, response, color='k', label='regression model')
    ax.scatter(x, y, edgecolor='k', facecolor='grey', alpha=0.7, label='sample data')
    ax.set_ylabel('succes', fontsize=14)
    ax.set_xlabel(i, fontsize=14)
    ax.legend(facecolor='white', fontsize=11)
    ax.set_title('$r^2= %.2f$' % r2, fontsize=18)

    fig.tight_layout()
    title = "graphs/" + i + '.png'
    plt.savefig(title)

    # for ii in df.corr():
    #     print(ii)
    print(i.replace("_", " "), "&", model.intercept_, "&", model.coef_[0])
    # X2 = sm.add_constant(X)
    # est = sm.OLS(y, X2)
    # est2 = est.fit()
    # print(est2.summary())
    # #print(pd.dataframe(zip(x.columns, model.coef_)))
    # print(pd.DataFrame(zip(x.columns, model.coef_)))

    # freg=f_regression(x,y)
    
    # p=freg[1]

    # print(p.round(3))



# for i in combined_items:
#     print("Test")
