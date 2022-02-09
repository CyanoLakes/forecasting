import matplotlib.pyplot as plt

SMALL_SIZE = 6
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def plot_timeseries(ax, result, horizon, name, variable, model):
    # Time series
    ax.plot(result.index, result['Obs'], 'k-', linewidth=1, markersize=3, label="Obs.")
    ax.plot(result.index, result['Pred'], 'b-', linewidth=0.9, label="1wk")
    if horizon == 'all':
        ax.plot(result.index, result['2wk'], 'g-', linewidth=0.9, label="2wk")
        ax.plot(result.index, result['4wk'], 'y-', linewidth=0.9, label="4wk")
    ax.set_title('%s' % (name), loc='center', y=0.85, x=0.5)

def plot_scatter(result, horizon, name):
    plt.figure()
    ax1 = result.plot.scatter('Obs', 'Pred', c='b', label='1wk')
    if horizon == 'all':
        result.plot.scatter('Obs', '2wk', c='g', ax=ax1, label='2wk')  # this needs to be shifted
        result.plot.scatter('Obs', '4wk', c='y', ax=ax1, label='4wk')  # this needs to be shifted
    plt.title(name)
    plt.ylabel('Pred. chl-a (ug/L)')
    plt.xlabel('Obs. chl-a (ug/L)')

def plot_decomposition(decomposed):
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