import itertools
import json

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.stats as statsimport
import statsmodels.api as sm


def precission(y, yhat):
    res = []
    for i in range(yhat):
        res.append((y[i] - yhat[i]) ** 2)

    return res


# define likelihood function
def MLERegression(params):  # Maximum likelihood estimation
    intercept, angular_coeff, dispersion = params[0], params[1], params[2]  # inputs are guesses at our parameters
    yhat = intercept + angular_coeff * x  # predictions next, we flip the Bayesian question
    # compute PDF of observed values normally distributed around mean (yhat)
    # with a standard deviation of dispersion
    negLL = -np.sum(scipy.stats.norm.logpdf(y, loc=yhat, scale=dispersion))  # return negative log likelihood
    return negLL


# generate data
np.random.seed(13)
N = 20
x = np.linspace(0, 20, N)
noise = np.random.normal(loc=0.0, scale=2.0, size=N)
y = 3 * x + 1 + noise
df = pd.DataFrame({'y': y, 'x': x})
df['constant'] = 0

plt.scatter(df.x, df.y)
plt.show()
# split features and target
X = df[['constant', 'x']]  # fit model and summarize
print('>>>>> 1-st method:')
print(sm.OLS(y, X).fit().summary())  # 1-й метод - МНК (статистическая оценка) - минимизировали сумму расстояний

# let’s start with some random coefficient guesses and optimize
# 2-й метод - метод максимимального правдоподобия (байесовская оценка) - максимизировали вероятность
# того, что точки принадлежат данному распределению
# guess = np.array([5, 5, 2])
print('>>>>> 2-st method')
guess_params = tuple(
    itertools.product(*[tuple(range(1,6))] * 3)
)
methods = (
    'Nelder-Mead',
    'Powell',
    'CG',
    'BFGS',
    'TNC',
    'SLSQP',
    # 'Newton-CG',  # ValueError: Jacobian is required for Newton-CG method
    # 'trust-constr',  # ValueError: ('Jacobian is required for trust region ', 'exact minimization.')
    # 'dogleg',  # ValueError: Jacobian is required for dogleg minimization
    # 'trust-ncg',  # ValueError: Jacobian is required for Newton-CG trust-region minimization
    # 'trust-exact',  # ValueError: Jacobian is required for trust region exact minimization.
    # 'trust-krylov',  # ValueError: ('Jacobian is required for trust region ', 'exact minimization.')
)

guess_samples_qnt = 5
result = {}
for method in methods:
    result[method] = {}

    for guess_param in guess_params:
        result[method].update({guess_param: []})

        for i in range(guess_samples_qnt):
            guess = np.array(guess_param)
            minimize_result = minimize(
                MLERegression,
                guess,
                method=method,
                # options={'disp': True},
            )
            result[method][guess_param].append(
                list(minimize_result.x)
                + ([minimize_result.nit] if hasattr(minimize_result, 'nit') else [])
                + ([minimize_result.nfev] if hasattr(minimize_result, 'nfev') else [])
            )

        result[method][str(guess_param)] = [
            sum(x)/guess_samples_qnt
            for x in zip(*result[method].pop(guess_param))
        ]

with open("simple_likelyhood.json", "w") as fp:
    json.dump(result, fp, indent=4)
