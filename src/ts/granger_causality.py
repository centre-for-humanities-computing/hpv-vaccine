"""Granger causality
"""

import ndjson
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller


def to_long_format(df, id, columns, val="val", val_id="val_id"):
    d = {"id": list(df[id])*len(columns),
         val: [ii for i in columns for ii in df[i]],
         val_id: [ii for i in columns for ii in [i]*len(df[i])]}
    df_long = pd.DataFrame(d)
    return df_long


def granger_causality(outcome, predictor, df, maxlag, **kwargs):
    data = df[[outcome, predictor]]
    gc_res = grangercausalitytests(data, maxlag, **kwargs)
    return gc_res


def adf(x, sign_level=0.01, raise_error=True):
    """
    Augmented Dickey-Fuller Test
    x (list): or list of lists
    """
    if isinstance(x[0], list):
        for i in x:
            adf(i)
        return "non_stationary"
    res = adfuller(x)
    if res[1] > sign_level:
        if raise_error:
            raise Exception("The time series is stationary")
        else:
            print("The time series is stationary")
            return "stationary"
    return "non_stationary"


def make_ts(init=1, alpha=0.3, beta1=1, beta2=-1, noise=True, size=100,
            plot=False):
    x = [init]
    for i in range(size-1):
        if len(x) == 1:
            x_ = alpha + beta1 * x[-1]
        elif noise:
            x_ = alpha + beta1 * x[-1] + beta2 * x[-2] + np.random.normal()
        else:
            x_ = alpha + beta1 * x[-1] + beta2 * x[-2]
        x.append(x_)
    if plot:
        fig = px.line(x)
        fig.show()
    return x


def make_caused_ts(x, alpha=-0, beta1=0.7, beta2=-0.5, beta3=2.1,
                   order=[-1, -2, -3], noise=True, size=100, plot=False):
    y = []
    for i, x_ in enumerate(x):
        if noise:
            y_ = alpha + beta1*x[i+order[0]] + beta2*x[i+order[1]] + \
                 beta3*x[i+order[2]] + np.random.normal()
        else:
            y_ = alpha + beta1*x[i+order[0]] + beta2*x[i+order[1]] + \
                 beta3*x[i+order[2]]
        y.append(y_)
    if plot:
        fig = px.line(y)
        fig.show()
    return y


def multiple_granger(df_outcome, df_predictor, maxlag, **kwargs):
    if df_outcome.shape[0] != df_predictor.shape[0]:
        raise ValueError("dataframes must have same length")

    res = []
    for col_o in df_outcome:
        for col_p in df_predictor:
            print(f"Checking column {col_o} ~ {col_p}")

            # make df for test
            df = df_outcome[[col_o]]
            df["pred"] = df_predictor[col_p].values
            gc = granger_causality(col_o, "pred", df, maxlag, **kwargs)
            res.append((col_o, col_p, gc))
    return res


def extract_significant(res, significance_level=0.05):
    significant = []
    for col_o, col_p, r in res:
        for lag in r:
            tests = ["ssr_ftest",
                    "ssr_chi2test",
                    "lrtest",
                    "params_ftest"]
            lowest = min(r[lag][0][t][1] for t in tests)
            if lowest < significance_level:
                significant.append([col_o, col_p, round(lowest, 4), lag, r])
    return significant


def main(df_predictors, df_outcome):
    '''
    Driver that compares each pair of timeseries provided.
    Individual timeseries are columns in respective dataframes. 
    
    all timeseries A (df_predictors) -> all timeseries B (df_outcomes)
    '''
    # Test for stationary timeseries
    print('[ADF test] {}'.format(df_predictors))
    for col_name, col_dat in df_predictors.iteritems():
        print(col_name, ":", adf(col_dat, raise_error=False))

    print('[ADF test] Testing timeseries in {}'.format(df_outcome))
    for col_name, col_dat in df_outcome.iteritems():
        print(col_name, ":", adf(col_dat, raise_error=False))

    # Multiple Granger
    res = multiple_granger(df_predictors, df_outcome, 5)
    sign = extract_significant(res, significance_level=0.01)

    # Format a table of results
    res_df = pd.DataFrame([i[0:4] for i in sign])
    res_df = res_df.rename({0: "Outcome",
                            1: "Predictor (Topic)",
                            2: "Significance (p<0.01)",
                            3: "maxlag"}, axis=1)
    return res_df
