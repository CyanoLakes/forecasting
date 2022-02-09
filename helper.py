import numpy as np

def trend(y0, y1):
    """
    Calculate linear trend from two weeks values
    """
    m = (y1 - y0) / 1  # dx = 1 week (weekly trend)
    return m

def get_seasonal_average(sa, dt, variable):
    """
    Return iso week from current index for next week
    """
    wk = dt.isocalendar()[1]  # forecast week
    if wk > sa.index[-1]:
        wk = wk - sa.index[-1]

    # Get seasonal average for forecast week
    return sa.loc[wk, variable]

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
