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
    wc = 0.5  # current inoculation
    ws = 0.3  # season
    wt = 0.1  # trend
    we = 0.1  # error

    w = 0.6
    # return (w * c) + (1 - w) * (s + t)
    # return (wc * c) + (ws * s) + (wt * t)
    # return (w * (c + t)) + (1 - w) * (s)

    e = c - p

    f = (w * c) + ((1 - w) * s)
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


def forecast(variable='chla_cyano',
             agg='week',
             model='naive',
             horizon=1,
             plot=False):
    """
    Forecasting function.
    :param model: model
    :param plot: create plots (True or False)
    :param horizon: forecast horizon (1, 2, 4 or all)
    :param variable: chla_med or chla_cyano
    :param agg: weekly or monthly aggregation of data
    :return: returns results dataframe
    """
    # 1. Read in file
    file_path = "/home/mark/Documents/Forecasting/CyanoLakes_all_stats.csv"
    csv = pd.read_csv(file_path)

    df = csv[[variable, 'name']]  # subset
    df.index = pd.to_datetime(csv['date'])  # make date index
    df = df.dropna(axis=0, how='any')
    df.replace(0, 1)  # replace zeros if exist

    # loop through unique names doing calculations
    names = df['name'].unique()  # get unique names

    # Pre-assign results array
    stats_results = pd.DataFrame(columns=names)

    for name in names:
        data = df[df['name'] == name]

        # Resample to weekly time-scale using mean and backfilling - moving average
        ma = data.resample('W').mean().bfill()

        # Centralized moving average
        cma = ma.rolling(window=2).mean()

        # Calculate trend equal to periodicity (12 months) moving average
        #trend = ma.rolling(window=52).mean()  # 52 weeks in a year

        # Subtract the trend (de-trend the time-series)

        # Calculate seasonal weekly averages from cma
        sa = cma.groupby(cma.index.isocalendar().week).mean()

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
                y2 = ma.loc[dt2, variable]  # next week (forecast)
            except KeyError:
                continue

            # Get seasonal average for forecast week
            s1 = get_seasonal_average(sa, dt2, variable)  # 1 wk forecast
            s2 = get_seasonal_average(sa, dt2 + datetime.timedelta(weeks=1), variable)  # 2 wk forecast
            s4 = get_seasonal_average(sa, dt2 + datetime.timedelta(weeks=3), variable)  # 4 wk forecast

            # Get trend
            # t = trend(y0, y1)
            t = change.loc[dt1].values[0]  # this weeks trend (from last weeks)
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
                    f1 = y1
                else:
                    f1 = y_f1[i-1]
                f1 = exponential_smoothing(y1, f1)

            if model == 'exponential smoothing error':
                # Set initial value
                if i == 0:
                    f1 = y1
                else:
                    f1 = y_f1[i-1]
                f1 = exponential_smoothing_err(y1, f1)

            if model == 'ets':
                if i == 0:
                    p1 = y1  # user current value
                else:
                    p1 = y_f1[i-1]
                f1 = ets(y1, p1, t, s1)

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
        stats_results.loc['rmse', name] = np.sqrt(np.sum(result['residual_sq']) / n)
        # result['residual_perc_1wk'] = abs((result['Obs'] - result['1wk']) / result['Obs'])
        #stats_results.loc['rsq_1wk', name] = np.square(result.corr().loc['Obs', '1wk'])
        #stats_results.loc['mae_1wk', name] = np.sum(result['residual_1wk']) / n
        #stats_results.loc['mape_1wk', name] = 100 * np.sum(result['residual_perc_1wk']) / n

        # Convert values to risk levels (how often does the risk level agree?)
        result['Obs_crl'] = result['Obs'].apply(parse_risk_level)
        result['Pred_crl'] = result['Pred'].apply(parse_risk_level)

        # Calculate CRL agreement in percentage 
        result['crl_agree'] = result['Obs_crl'] == result['Pred_crl']
        stats_results.loc['crl', name] = 100 * round(np.sum(result['crl_agree']) / n, 3)

        if horizon == 'all':
            # n = n - 1 week
            result['residual_2wk'] = abs(result['Obs'] - result['2wk'])
            result['residual_sq_2wk'] = np.square(result['Obs'] - result['2wk'])
            stats_results.loc['rmse_2wk', name] = np.sqrt(np.sum(result['residual_sq_2wk']) / (n - 1))
            # n = n - 3 weeks
            result['residual_4wk'] = abs(result['Obs'] - result['4wk'])
            result['residual_sq_4wk'] = np.square(result['Obs'] - result['4wk'])
            stats_results.loc['rmse_4wk', name] = np.sqrt(np.sum(result['residual_sq_4wk']) / (n - 3))
            result['2wk_crl'] = result['2wk'].apply(parse_risk_level)
            result['4wk_crl'] = result['4wk'].apply(parse_risk_level)
            result['2wk_crl_agree'] = result['Obs_crl'] == result['2wk_crl']
            stats_results.loc['crl_2wk', name] = 100 * round(np.sum(result['2wk_crl_agree']) / (n - 1), 3)
            result['4wk_crl_agree'] = result['Obs_crl'] == result['4wk_crl']
            stats_results.loc['crl_4wk', name] = 100 * round(np.sum(result['4wk_crl_agree']) / (n - 3), 3)

        # Charts
        if plot:
            # Time series
            plt.figure()
            result['Obs'].plot(style='k.-', label="Obs.")
            result['Pred'].plot(style='b-', label="1wk")
            if horizon == 'all':
                result['2wk'].plot(style='g-', label="2wk")
                result['4wk'].plot(style='y-', label="4wk")
            result['residual'].plot(style='r+', label="residual")
            plt.legend()
            plt.ylabel('Chl-a (ug/L)')
            plt.title('%s %s %s' % (name, variable, model))

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
            # plt.figure()
            # change.plot()
            # ad.plot()
            # data['chla_cyano'].plot()
            # ma['chla_cyano'].plot()
            # cma['chla_cyano'].plot()
            # sa['chla_cyano'].plot()

            plt.show()

    # result = seasonal_decompose(ma['chla_cyano'], model="add")
    # result.plot()
    #print('Variable: %s Agg: %s Model: %s Horizon: %s' % (variable, agg, model, horizon))
    #print(stats_results)
    return stats_results, names
    # stats_results.to_excel("/home/mark/PycharmProjects/Forecasting/stats_results.xlsx")


if __name__ == "__main__":

    # models:  naive, moving average, seasonal naive, exponential smoothing, ets
    # aggregation: weekly (W) or monthly (M)
    # forecast horizon: 1 or all( 1, 2, and 4)
    # plot (create plots)

    n, names = forecast(variable='chla_cyano', agg='W', model='naive', horizon=1, plot=False)
    ma, names = forecast(variable='chla_cyano', agg='W', model='moving average', horizon=1, plot=False)
    sn, names = forecast(variable='chla_cyano', agg='W', model='seasonal naive', horizon=1, plot=False)
    es, names = forecast(variable='chla_cyano', agg='W', model='exponential smoothing', horizon=1, plot=False)
    ets, names = forecast(variable='chla_cyano', agg='W', model='ets', horizon=1, plot=True)
    #forecast(variable='chla_med', agg='W', model='naive', horizon=1, plot=False)

    # Can combine results here
    index = ['naive', 'moving average', 'seasonal naive', 'exponential smoothing', 'ets']
    df = pd.DataFrame(columns=names, index=index)
    df.loc['naive', :] = n.loc['rmse', :]
    df.loc['moving average', :] = ma.loc['rmse', :]
    df.loc['seasonal naive', :] = sn.loc['rmse', :]
    df.loc['exponential smoothing', :] = es.loc['rmse', :]
    df.loc['ets', :] = ets.loc['rmse', :]

    print(df)

