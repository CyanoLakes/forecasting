import pandas as pd
import settings
from forecast import forecast

def baseline():
    # Call all baseline models
    n, names = forecast(variable='chla_cyano', model='naive', horizon=1, plot=False)
    ma, names = forecast(variable='chla_med', model='moving average', horizon=1, plot=False)
    sn, names = forecast(variable='chla_med', model='seasonal naive', horizon=1, plot=False)
    es, names = forecast(variable='chla_med', model='exponential smoothing', horizon=1, plot=False)
    taes, names = forecast(variable='chla_med', model='trend adjusted exponential smoothing', horizon=1, plot=False)
    ets, names = forecast(variable='chla_cyano', model='ets', horizon=1, plot=True)

    # Combine results into one file
    index = ['na', 'ma', 'sn', 'es', 'taes', 'ets']
    df = pd.DataFrame(columns=names, index=index)
    df.loc['na', :] = n.loc['rmse', :]
    df.loc['ma', :] = ma.loc['rmse', :]
    df.loc['sn', :] = sn.loc['rmse', :]
    df.loc['es', :] = es.loc['rmse', :]
    df.loc['taes', :] = taes.loc['rmse', :]
    df.loc['ets', :] = ets.loc['rmse', :]

    # Write results to file
    df.to_excel(settings.root + settings.output_path + "Baseline.xlsx")