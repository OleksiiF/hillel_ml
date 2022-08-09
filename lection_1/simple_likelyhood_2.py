# import libraries
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as statsimport
# import pymc3 as pm3
# import numdifftools as ndt
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel


def precission(y, yhat):
    res = []
    for i in range(yhat):
        res.append((y[i] - yhat[i]) ** 2)

    return res


# define likelihood function
def MLERegression(params):
    intercept, beta, sd = (params[0], params[1], params[2])  # inputs are guesses at our parameters
    yhat = intercept + beta*x  # predictions# next, we flip the Bayesian question
    # compute PDF of observed values normally distributed around mean (yhat)
    # with a standard deviation of sd
    negLL = -np.sum(scipy.stats.norm.logpdf(y, loc=yhat, scale=sd))  # return negative LL
    return(negLL)


# generate data
N = 20
x = np.linspace(0, 20, N)
noise = np.random.normal(loc=0.0, scale=2.0, size=N)
y = 3 * x + 1 + noise
df = pd.DataFrame({'y': y, 'x': x})
df['constant'] = 0

print(df.head(15))

plt.scatter(df.x, df.y)
plt.show()
#  split features and target
X = df[['constant', 'x']]  # fit model and summarize
print('1-st method:')
print(sm.OLS(y, X).fit().summary())  # 1-й метод - МНК (статистическая оценка) - минимизировали сумму расстояний
#
# let’s start with some random coefficient guesses and optimize
# 2-й метод - метод максимимального правдоподобия (байесовская оценка) - максимизировали вероятность
# того, что точки принадлежат данному распределению
guess = np.array([5, 5, 2])
print('2-st method:')
results = minimize(
    MLERegression,
    guess,
    method='Nelder-Mead',
    options={'disp': True}
)
