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

def moving_average_seasonal_error_adjusted(ma, sc, sn, horizon=1):
    """
    Simple error adjusted seasonal model
    :param ma: moving average
    :param sc: seasonal average (current)
    :param sn: seasonal average (next)
    :return: forecast value
    """
    # Weight seasonal component higher when forecasting ahead
    if horizon == 4:
        a = 0.6  # alpha weighting factor
    elif horizon == 2:
        a = 0.7
    else:
        a = 0.8
    anom = ma - sc  # seasonal anomaly (difference between moving average and seasonal average)
    f = (a * ma) + (1 - a) * (sn + anom)
    if f < 0:
        return 0
    return f

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

