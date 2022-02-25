from forecast import forecast
from baseline import baseline


if __name__ == "__main__":

    # Set model and other options here
    variable = 'chla_cyano'  # can be either 'chla_cyano' or 'chla_med'
    model = 'masea'  # select model, can be
    horizon = 'all'  # use 1, 2 or all (forecast horizon) 1-week, 2-week, 4-week or all
    plot = True  # draw plots

    forecast(variable=variable, model=model, horizon=horizon, plot=plot)

    # If you would like to compare baseline models, uncomment below
    # baseline()