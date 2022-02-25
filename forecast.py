import pandas as pd
import numpy as np
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import settings
from helper import *
from charts import *
from models import *


def forecast(variable='chla_cyano',
             model='naive',
             horizon=1,
             plot=False):
    """
    Forecasting function.
    :param model: model - name of the model can be 'naive', 'moving average', 'seasonal naive',
    'exponential smoothing', 'trend adjusted exponential smoothing', 'exponential smoothing err',
    'ets', 'masea'
    :param plot: create plots (True or False)
    :param horizon: forecast horizon (1, 2, 4 or 'all')
    :param variable: chla_med or chla_cyano
    :return: returns results dataframe
    """

    print('Variable: %s Model: %s Horizon: %s' % (variable, model, horizon))

    # 1. Read in file
    file_path = settings.ROOT + settings.INPUT_FILE
    csv = pd.read_csv(file_path)

    # Optionally plot all raw timeseries
    if plot:
        plot_raw_timeseries(csv)

    df = csv[[variable, 'name']]  # subset
    df.index = pd.to_datetime(csv['date'])  # make date index
    df = df.dropna(axis=0, how='any')
    # df.replace(0, 0.00001)  # replace zeros if exist

    # loop through unique names doing calculations
    names = df['name'].unique()  # get unique names

    # Pre-assign results array
    results = pd.DataFrame(columns=names)
    stats = pd.DataFrame(columns=names)

    # Set up subplots
    fig, axs = plt.subplots(3, 5, constrained_layout=True, sharex=True)

    for name, ax in zip(names, axs.flat):
        data = df[df['name'] == name]

        # Get some generic stats
        stats.loc[variable, name] = data[variable].mean()
        stats.loc['count', name] = len(data)
        stats.loc['start', name] = data.index.min()
        stats.loc['end', name] = data.index.max()

        # Resample to weekly time-scale using mean and backfilling - moving average
        ma = data.resample('W').mean().bfill()

        # Centralized moving average
        cma = ma.rolling(window=2).mean()

        # 2. Classical ETS decomposition
        # Calculate trend equal to periodicity (12 months) moving average
        trend = ma.rolling(window=52).mean()  # 52 weeks in a year
        detrended = ma - trend  # Subtract the trend (de-trend the time-series)
        seasonal = detrended.groupby(detrended.index.isocalendar().week).mean()  # Seasonal values from detrended data (weekly)
        seasonal_series = ma.apply(lambda x: seasonal.loc[x.index.isocalendar().week, variable])  # make seasonal time series
        seasonal_series.index = ma.index
        remainder = ma - trend - seasonal_series
        reconstruct = remainder + trend + seasonal_series
        columns = ['reconstructed', 'trend', 'seasonality', 'remainder']
        decomposed = pd.DataFrame(columns=columns, index=ma.index)
        decomposed.loc[:, 'original'] = ma.loc[:, variable]
        decomposed.loc[:, 'reconstructed'] = reconstruct.loc[:, variable]
        decomposed.loc[:, 'trend'] = trend.loc[:, variable]
        decomposed.loc[:, 'seasonality'] = seasonal_series.loc[:, variable]
        decomposed.loc[:, 'remainder'] = remainder.loc[:, variable]

        if variable == 'chla_cyano':
            cycma = cma.where(cma < 1, 1)  # where > 0 make 1 else 0
            cyct = cycma.groupby(cycma.index.isocalendar().week).count()  # weekly count
            cysum = cycma.groupby(cycma.index.isocalendar().week).sum()  # weekly sum
            cyprob = cysum / cyct

        # pre-assign lists (speed)
        y = list()  # actual chl
        y_f1 = list()  # 1 wk forecast
        y_f2 = list()  # 2 wk forecast
        y_f4 = list()  # 4 wk forecast
        dates = list()
        taes = list()  # for trend adjusted exponential smoothing

        # Loop through last 1 year of data calculating weekly forecast values
        indices = ma[ma.index > (ma.index[-1] - datetime.timedelta(365))].index
        i = 0
        for dt in indices:
            dt1 = dt + datetime.timedelta(weeks=1)  # this week
            dt2 = dt + datetime.timedelta(weeks=2)  # 1 week forecast

            # Get actual values
            try:
                y0 = ma.loc[dt, variable]  # last week
                y1 = ma.loc[dt1, variable]  # this week
                y2 = ma.loc[dt2, variable]  # next week (forecast week)
            except KeyError:
                continue

            # Get trend
            t = y1 - y0  # change since last week

            # Compute forecasts
            if model == 'naive':
                # forecast is current value
                f1 = y1

            if model == 'moving average':
                # forecast is mean of last 2 values
                f1 = (y0 + y1) / 2

            if model == 'seasonal naive':
                # forecast is seasonal value
                f1 = s1

            if model == 'exponential smoothing':
                # Set initial value
                if i == 0:
                    f0 = y1
                else:
                    f0 = y_f1[i-1]
                f1 = exponential_smoothing(y1, f0)

            if model == 'exponential smoothing error':
                # Set initial value
                if i == 0:
                    f0 = y1
                else:
                    f0 = y_f1[i-1]
                f1 = exponential_smoothing_err(y1, f0)

            if model == 'trend adjusted exponential smoothing':
                # Set initial value
                if i == 0:
                    f0 = y1
                    t0 = 0
                else:
                    f0 = y_f1[i - 1]
                    t0 = taes[i - 1]
                f1, t1 = trend_adjusted_exponential_smoothing(y1, f0, t0)
                taes.append(t1)

            if model == 'ets':
                if i == 0:
                    p1 = y1  # use current value
                else:
                    p1 = y_f1[i-1]
                f1 = ets(y1, p1, t, s1)

            if model == 'logical decomposition':
                # Set initial value
                if i == 0:
                    f0 = y1
                else:
                    f0 = y_f1[i-1]
                f1 = logical_decomposition(y0, y1, cp1, s1, t, f0)

            if model == "masea":
                # Get seasonal average for forecast week
                mcma = cma[cma.index <= dt1]  # include all data up to current value (no future data)
                sa = mcma.groupby(mcma.index.isocalendar().week).mean()
                s0 = get_seasonal_average(sa, dt1, variable)  # current value
                s1 = get_seasonal_average(sa, dt2, variable)  # 1 wk forecast
                s2 = get_seasonal_average(sa, dt2 + datetime.timedelta(weeks=1), variable)  # 2 wk forecast
                s4 = get_seasonal_average(sa, dt2 + datetime.timedelta(weeks=3), variable)  # 4 wk forecast

                mva = (y0 + y1) / 2  # moving average
                f1 = moving_average_seasonal_error_adjusted(mva, s0, s1)


            dates.append(dt2)  # 1 wk forecast dates
            y.append(y2)  # Chl-a observed (actual value at forecast date)
            y_f1.append(f1)

            if horizon == 'all':
                f2 = moving_average_seasonal_error_adjusted(mva, s0, s2, horizon=2)
                # f2 = ets2wk(y1, t, s2)
                y_f2.append(f2)  # 2 week forecast value
                f4 = moving_average_seasonal_error_adjusted(mva, s0, s4, horizon=4)
                # f4 = ets4wk(y1, t, s4)
                y_f4.append(f4)  # 4 week forecast value

            i += 1

        # Append results
        if horizon == 1:
            result = pd.DataFrame(list(zip(y, y_f1)), index=dates, columns=['Obs', 'Pred'])

        if horizon == 'all':
            result_wk1 = pd.DataFrame(list(zip(y, y_f1)), index=dates, columns=['Obs', 'Pred'])
            dates2 = [date + datetime.timedelta(weeks=1) for date in dates]
            dates4 = [date + datetime.timedelta(weeks=3) for date in dates]
            result_wk2 = pd.DataFrame(y_f2, index=dates2, columns=['2wk',])
            result_wk4 = pd.DataFrame(y_f4, index=dates4, columns=['4wk', ])
            result = pd.concat([result_wk1, result_wk2, result_wk4], axis=1)

        # Calculate RMSE performance metric
        n = len(dates)
        result['residual'] = abs(result['Obs'] - result['Pred'])
        result['residual_sq'] = np.square(result['Obs'] - result['Pred'])
        results.loc['rmse', name] = np.sqrt(np.sum(result['residual_sq']) / n)

        # Alternative performance metrics
        # result['residual_lsq'] = np.square(np.log(result['Obs']) - np.log(result['Pred']))
        # results.loc['mae', name] = np.sum(result['residual']) / n
        # results.loc['rmsle', name] = np.exp(np.sqrt(np.sum(result['residual_lsq']) / n))
        # result['residual_perc'] = abs((result['Obs'] - result['Pred'])) / result['Obs']
        # results.loc['rsq_1wk', name] = np.square(result.corr().loc['Obs', '1wk'])
        #results.loc['mape', name] = 100 * np.sum(result['residual_perc']) / n

        print("%s rmse: %s" % (name, str(results.loc['rmse', name])))

        # CRL agreement
        if variable == 'chla_cyano':
            result['Obs_crl'] = result['Obs'].apply(parse_risk_level)
            result['Pred_crl'] = result['Pred'].apply(parse_risk_level)
            result['crl_agree'] = result['Obs_crl'] == result['Pred_crl']
            result['crl_high_agree'] =  np.logical_and(result['Obs_crl'] >= 2, result['Pred_crl'] >= 2)
            n_high = np.sum((result['Obs_crl'] >= 2))
            results.loc['crl', name] = 100 * round(np.sum(result['crl_agree']) / n, 3)
            if n_high >= 1:
                results.loc['crl_high', name] = 100 * round(np.sum(result['crl_high_agree']) / n_high, 3)
            results.loc['n_crl_high', name] = n_high

        # Trophic state agreement
        if variable == 'chla_med':
            result['Obs_ts'] = result['Obs'].apply(parse_trophic_state)
            result['Pred_ts'] = result['Pred'].apply(parse_trophic_state)
            result['ts_agree'] = result['Obs_ts'] == result['Pred_ts']
            results.loc['ts', name] = 100 * round(np.sum(result['ts_agree']) / n, 3)

        if horizon == 'all':
            # n = n - 1 week
            result['residual_2wk'] = abs(result['Obs'] - result['2wk'])
            result['residual_sq_2wk'] = np.square(result['Obs'] - result['2wk'])
            results.loc['rmse_2wk', name] = np.sqrt(np.sum(result['residual_sq_2wk']) / (n - 1))
            # n = n - 3 weeks
            result['residual_4wk'] = abs(result['Obs'] - result['4wk'])
            result['residual_sq_4wk'] = np.square(result['Obs'] - result['4wk'])
            results.loc['rmse_4wk', name] = np.sqrt(np.sum(result['residual_sq_4wk']) / (n - 3))
            
            if variable == 'chla_cyano':
                result['2wk_crl'] = result['2wk'].apply(parse_risk_level)
                result['4wk_crl'] = result['4wk'].apply(parse_risk_level)
                result['2wk_crl_agree'] = result['Obs_crl'] == result['2wk_crl']
                result['2wk_crl_high_agree'] =  np.logical_and(result['Obs_crl'] >= 2, result['2wk_crl'] >= 2)
                results.loc['crl_2wk', name] = 100 * round(np.sum(result['2wk_crl_agree']) / (n - 1), 3)
                result['4wk_crl_agree'] = result['Obs_crl'] == result['4wk_crl']
                result['4wk_crl_high_agree'] =  np.logical_and(result['Obs_crl'] >= 2, result['4wk_crl'] >= 2)
                results.loc['crl_4wk', name] = 100 * round(np.sum(result['4wk_crl_agree']) / (n - 3), 3)
                if n_high >= 1:
                    results.loc['crl_high_2wk', name] = 100 * round(np.sum(result['2wk_crl_high_agree']) / n_high, 3)
                    results.loc['crl_high_4wk', name] = 100 * round(np.sum(result['4wk_crl_high_agree']) / n_high, 3)

            if variable == 'chla_med':
                result['2wk_ts'] = result['2wk'].apply(parse_trophic_state)
                result['4wk_ts'] = result['4wk'].apply(parse_trophic_state)
                result['2wk_ts_agree'] = result['Obs_ts'] == result['2wk_ts']
                results.loc['ts_2wk', name] = 100 * round(np.sum(result['2wk_ts_agree']) / (n - 1), 3)
                result['4wk_ts_agree'] = result['Obs_ts'] == result['4wk_ts']
                results.loc['ts_4wk', name] = 100 * round(np.sum(result['4wk_ts_agree']) / (n - 3), 3)

        # Charts
        if plot:
            plot_timeseries(ax, result, horizon, name, variable, model)
            plot_scatter(result, horizon, name)
            plot_decomposition(name, decomposed)

    # Print results
    print('Mean rmse: %s' % str(results.loc['rmse',:].mean()))
    if horizon == 'all':
        print('Mean rmse 2wk: %s' % str(results.loc['rmse_2wk',:].mean()))
        print('Mean rmse 4 wk: %s' % str(results.loc['rmse_4wk',:].mean()))

    # Save results to excel
    stats.to_excel(settings.ROOT + settings.OUTPUT_PATH + "timeseries_stats.xlsx")
    results.to_excel(settings.ROOT + settings.OUTPUT_PATH + model + "_rmse_results.xlsx")

    if plot:
        format(variable, axs)

    results.to_excel(settings.ROOT + settings.OUTPUT_PATH + variable + ".xlsx")

    return results, names