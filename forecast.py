import pandas as pd
import numpy as np
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


def trend(y0, y1):
    """
    Calculate linear trend from two weeks values
    """
    m = (y1 - y0) / 1  # dx = 1 week (weekly trend)
    return m


def exponential_smoothing(c, p):
    """
    Exponential smoothing function
    :param c - actual value
    :param p - predicted value
    :returns f - forecast value
    """
    a = 0.7  # alpha ranges from 0 to 1
    return (a * c) + (1 - a) * p


def trend_adjusted_exponential_smoothing(c, p, t):
    """
    Exponential smoothing function
    :param c - current actual value for t
    :param p - predicted forecast value for t
    :param t - previous trend
    :returns f - forecast value
    """
    A = 0.5  # alpha ranges from 0 to 1
    B = 0.5  # beta  ranges from 0 to 1 (trend adjustment factor)

    # Exp smoothing
    f = (A * c) + (1 - A) * p

    # Trend (forecast trend)
    ft = B * (f - p) + (1 - B) * t

    # Trend adjusted forecast
    taf = f + ft
    if taf < 0:
        taf = 0

    return taf, ft


def exponential_smoothing_err(c, p):
    """
    Exponential smoothing function
    :param c - actual value
    :param p - predicted value
    :returns f - forecast value
    """
    a = 0.4  # alpha can be any value (typically between 0 and 1)
    e = c - p  # residual error
    f = c + (a * e)

    # constrain to positive values
    if f < 1:
        return 1
    return f


def get_seasonal_average(sa, dt, variable):
    """
    Return iso week from current index for next week
    """
    wk = dt.isocalendar()[1]  # forecast week
    if wk == 54:
        wk = 1
    if wk == 55:
        wk = 2

    # Get seasonal average for forecast week
    return sa.loc[wk, variable]


def ets(c, p, t, s):
    """
    Forecast model
    :param - c current value
    :param - p last predicted value
    :param - t trend
    :param - s seasonal value

    0.5 weighting for seasonal value
    0.5 weighting for current value times trend
    """
    w = 0.6


    # 1. Forecast is weighting of seasonal and current inoculation
    f = (w * c) + (1 - w) * s

    # 2. Forecast is weighting of seasonal, current inoculation plus trend adjustment
    # f = (w * (c + t)) + (1 - w) * s  # not very good
    # f = (w * c) + ((1 - w) * s) + t  # trend too strong
    # f = (w * c) + (1 - w) * (s + t)  # not great

    # 3. Forecast is seasonal, current plus error of last forecast
    e = c - p  # difference of previous forecast and actual value
    #f = (w * (c + e)) + ((1 - w) * s)  # poor
    #f = (w * c) + ((1 - w) * s) + e  # too jumpy

    # 4.
    # wc = 0.5  # current inoculation
    # ws = 0.3  # season
    # wt = 0.1  # trend
    # we = 0.1  # error

    # return (wc * c) + (ws * s) + (wt * t)

    if f < 1:
        f = 1

    return f

    # e = c - p   # error of previous prediction
    # if c > 0:
    #     ew = abs((c - p)) / c
    #     if ew > 1:
    #         wc = 0.1
    #         we = 0.5
    #         print("We >>")
    #
    # f = (wc * c) + (wt * t) + (ws * s) + (we * e)
    # if f < 1:
    #     f = 1
    #
    # return f
    # Add exponential smoothing
    #return exponential_smoothing(c, f)

    # return (w * c) + (1 - w) * (s)
    # return s


def logical_decomposition(y1, y0, cp1, sa, t, f0):
    """Custom model
    y1 - current value
    y0 - last week value
    cp1 - seasonal cyanobacteria probability of forecast week
    t - trend
    f0 - forecast for this week (the last forecast)
    """

    # 1. Moving average
    ma = (y1 + y0) / 2

    # 2. Error of last forecast
    e = y1 - f0

    # 3. Weight MA by cyano probability
    #f = (ma * cp1)  # + e
    f = ma + e

    # weighted current and seasonal average with error
    a = 0.7
    f = (a * y1) + (1 - a) * sa



    if f < 1:
        return 1

    return f


def ets2wk(c, t, s):
    w = 0.5  # current inoculation weighting factor
    return (w * c) + (1 - w) * s


def ets4wk(c, t, s):
    w = 0.3  # current inoculation
    return (w * c) + (1 - w) * s


def parse_risk_level(x):
    """Returns risk level for chl value
    :returns
    0 = low
    1 = med
    2 = high
    3 = very high
    """
    if np.isnan(x):
        return np.nan
    if x > 100:
        return 3
    if x > 50:
        return 2
    if x > 10:
        return 1
    return 0


def parse_trophic_state(x):
     if np.isnan(x):
        return np.nan
     if x > 50:
         return 3  #Hyper
     if x > 20:
         return 2  #Eu
     if x > 10:
         return 1 #Meso
     return 0 # Oligo


def forecast(variable='chla_cyano',
             model='naive',
             horizon=1,
             plot=False):
    """
    Forecasting function.
    :param model: model - name of the model
    :param plot: create plots (True or False)
    :param horizon: forecast horizon (1, 2, 4 or all)
    :param variable: chla_med or chla_cyano
    :return: returns results dataframe
    """
    print('Variable: %s Model: %s Horizon: %s' % (variable, model, horizon))

    # 1. Read in file
    file_path = "/home/mark/PycharmProjects/forecasting/CyanoLakes_chl_stats.csv"
    csv = pd.read_csv(file_path)

    df = csv[[variable, 'name']]  # subset
    df.index = pd.to_datetime(csv['date'])  # make date index
    df = df.dropna(axis=0, how='any')
    # df.replace(0, 0.00001)  # replace zeros if exist

    # loop through unique names doing calculations
    names = df['name'].unique()  # get unique names

    # Pre-assign results array
    results = pd.DataFrame(columns=names)
    stats = pd.DataFrame(columns=names)

    for name in names:
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

        # Calculate trend equal to periodicity (12 months) moving average
        trend = ma.rolling(window=52).mean()  # 52 weeks in a year

        # Subtract the trend (de-trend the time-series)
        detrended = ma - trend

        # Seasonal values from detrended data (weekly)
        seasonal = detrended.groupby(detrended.index.isocalendar().week).mean()

        # make seasonal time series
        seasonal_series = ma.apply(lambda x: seasonal.loc[x.index.isocalendar().week, variable])
        seasonal_series.index = ma.index

        # Remainder
        remainder = ma - trend - seasonal_series

        reconstruct = remainder + trend + seasonal_series
        columns = ['reconstructed', 'trend', 'seasonality', 'remainder']
        decomposed = pd.DataFrame(columns=columns, index=ma.index)
        decomposed.loc[:, 'original'] = ma.loc[:, 'chla_cyano']
        decomposed.loc[:, 'reconstructed'] = reconstruct.loc[:, 'chla_cyano']
        decomposed.loc[:, 'trend'] = trend.loc[:, 'chla_cyano']
        decomposed.loc[:, 'seasonality'] = seasonal_series.loc[:, 'chla_cyano']
        decomposed.loc[:, 'remainder'] = remainder.loc[:, 'chla_cyano']


        # Calculate seasonal weekly averages from cma
        sa = cma.groupby(cma.index.isocalendar().week).mean()

        if variable == 'chla_cyano':
            cycma = cma.where(cma < 1, 1)  # where > 0 make 1 else 0
            cyct = cycma.groupby(cycma.index.isocalendar().week).count()  # weekly count
            cysum = cycma.groupby(cycma.index.isocalendar().week).sum()  # weekly sum
            cyprob = cysum / cyct

        # Subtract seasonal signal from resampled time-series (to get anomalies)
        subtract_array = ma.apply(lambda x: sa.loc[x.index.isocalendar().week, variable])
        subtract_array.index = ma.index
        anomalies = ma - subtract_array

        # Calculate change (differential) as the trend + error
        change = anomalies - anomalies.shift(2)
        ad = anomalies.diff()  # equivalent to above

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

            # Get seasonal average for forecast week
            s1 = get_seasonal_average(sa, dt2, variable)  # 1 wk forecast
            s2 = get_seasonal_average(sa, dt2 + datetime.timedelta(weeks=1), variable)  # 2 wk forecast
            s4 = get_seasonal_average(sa, dt2 + datetime.timedelta(weeks=3), variable)  # 4 wk forecast

            # Get seasonal probability
            if variable == 'chla_cyano':
                cp1 = get_seasonal_average(cyprob, dt2, variable)

            # Get trend
            # t = trend(y0, y1)
            #t = change.loc[dt1].values[0]  # this weeks trend (from last weeks)
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

            dates.append(dt2)  # 1 wk forecast dates
            y.append(y2)  # Chl-a observed (actual value at forecast date)
            y_f1.append(f1)

            if horizon == 'all':
                f2 = ets2wk(y1, t, s2)
                y_f2.append(f2)  # 2 week forecast value
                f4 = ets4wk(y1, t, s4)
                y_f4.append(f4)  # 4 week forecast value

            i += 1

        # Append results
        # when appending 2wk and 4wk forecast shift the index date... (otherwise it doesn't don't align)
        if horizon == 1:
            result = pd.DataFrame(list(zip(y, y_f1)), index=dates, columns=['Obs', 'Pred'])

        if horizon == 'all':
            result_wk1 = pd.DataFrame(list(zip(y, y_f1)), index=dates, columns=['Obs', 'Pred'])
            dates2 = [date + datetime.timedelta(weeks=1) for date in dates]
            dates4 = [date + datetime.timedelta(weeks=3) for date in dates]
            result_wk2 = pd.DataFrame(y_f2, index=dates2, columns=['2wk',])
            result_wk4 = pd.DataFrame(y_f4, index=dates4, columns=['4wk', ])
            result = pd.concat([result_wk1, result_wk2, result_wk4], axis=1)

        # Calculate performance (we are only using rmse which uses arithmetic mean)
        n = len(dates)
        result['residual'] = abs(result['Obs'] - result['Pred'])
        result['residual_sq'] = np.square(result['Obs'] - result['Pred'])
        # result['residual_lsq'] = np.square(np.log(result['Obs']) - np.log(result['Pred']))
        results.loc['rmse', name] = np.sqrt(np.sum(result['residual_sq']) / n)
        # results.loc['mae', name] = np.sum(result['residual']) / n
        # results.loc['rmsle', name] = np.exp(np.sqrt(np.sum(result['residual_lsq']) / n))
        # result['residual_perc'] = abs((result['Obs'] - result['Pred'])) / result['Obs']
        # results.loc['rsq_1wk', name] = np.square(result.corr().loc['Obs', '1wk'])
        #results.loc['mape', name] = 100 * np.sum(result['residual_perc']) / n

        # CRL agreement
        if variable == 'chla_cyano':
            result['Obs_crl'] = result['Obs'].apply(parse_risk_level)
            result['Pred_crl'] = result['Pred'].apply(parse_risk_level)
            result['crl_agree'] = result['Obs_crl'] == result['Pred_crl']
            results.loc['crl', name] = 100 * round(np.sum(result['crl_agree']) / n, 3)

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
            result['2wk_crl'] = result['2wk'].apply(parse_risk_level)
            result['4wk_crl'] = result['4wk'].apply(parse_risk_level)
            result['2wk_crl_agree'] = result['Obs_crl'] == result['2wk_crl']
            results.loc['crl_2wk', name] = 100 * round(np.sum(result['2wk_crl_agree']) / (n - 1), 3)
            result['4wk_crl_agree'] = result['Obs_crl'] == result['4wk_crl']
            results.loc['crl_4wk', name] = 100 * round(np.sum(result['4wk_crl_agree']) / (n - 3), 3)

        # Charts
        if plot:
            # Time series
            # plt.figure()
            # result['Obs'].plot(style='k.-', label="Obs.")
            # result['Pred'].plot(style='b-', label="1wk")
            # if horizon == 'all':
            #     result['2wk'].plot(style='g-', label="2wk")
            #     result['4wk'].plot(style='y-', label="4wk")
            # result['residual'].plot(style='r+', label="residual")
            # plt.legend()
            # plt.ylabel('Chl-a (ug/L)')
            # plt.title('%s %s %s' % (name, variable, model))

            # Scatter
            # plt.figure()
            # ax1 = result.plot.scatter('Obs', '1wk', c='b', label='1wk')
            # if horizon == 'all':
            #     result.plot.scatter('Obs', '2wk', c='g', ax=ax1, label='2wk')  # this needs to be shifted
            #     result.plot.scatter('Obs', '4wk', c='y', ax=ax1, label='4wk')  # this needs to be shifted
            # plt.title(name)
            # plt.ylabel('Pred. chl-a (ug/L)')
            # plt.xlabel('Obs. chl-a (ug/L)')

            # Seasonal decomposition
            plt.figure()
            # change.plot()
            # ad.plot()
            # data['chla_cyano'].plot()
            # ma['chla_cyano'].plot()
            # cma['chla_cyano'].plot()
            # sa['chla_cyano'].plot()
            # ma.plot()
            # trend.plot()
            # seasonal_series.plot()
            # remainder.plot()
            decomposed.plot()


            plt.show()


    # result = seasonal_decompose(ma['chla_cyano'], model="add")
    # result.plot()
    print(results)
    stats.to_excel("/home/mark/Documents/Forecasting/stats.xlsx")
    return results, names
    # results.to_excel("/home/mark/PycharmProjects/Forecasting/results.xlsx")


if __name__ == "__main__":
    # n, names = forecast(variable='chla_cyano', model='naive', horizon=1, plot=False)
    # ma, names = forecast(variable='chla_med', model='moving average', horizon=1, plot=False)
    # sn, names = forecast(variable='chla_med', model='seasonal naive', horizon=1, plot=False)
    # es, names = forecast(variable='chla_med', model='exponential smoothing', horizon=1, plot=False)
    # taes, names = forecast(variable='chla_med', model='trend adjusted exponential smoothing', horizon=1, plot=False)
    # ets, names = forecast(variable='chla_cyano', model='ets', horizon=1, plot=True)
    lgdcpstn, names = forecast(variable='chla_cyano', model='logical decomposition', horizon=1, plot=True)
    #forecast(variable='chla_med', model='naive', horizon=1, plot=False)

    # Can combine results here
    # index = ['na', 'ma', 'sn', 'es', 'taes', 'ets']
    # df = pd.DataFrame(columns=names, index=index)
    # df.loc['na', :] = n.loc['rmse', :]
    # df.loc['ma', :] = ma.loc['rmse', :]
    # df.loc['sn', :] = sn.loc['rmse', :]
    # df.loc['es', :] = es.loc['rmse', :]
    # df.loc['taes', :] = taes.loc['rmse', :]
    # df.loc['ets', :] = ets.loc['rmse', :]

    # df.to_excel("/home/mark/PycharmProjects/forecasting/chl_results.xlsx")

    # print(df)
    # print(es)

