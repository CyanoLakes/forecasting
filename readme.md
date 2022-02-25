## Simple univariate forecasting models

This code provides comparison of simple baseline univariate forecasting models
used to forecast cyanobacteria blooms and chlorophyll-a concentrations in lakes. 
It uses data from the CyanoLakes Application from satellite remote sensing data. 
Forecasts are produced using 1 year of hold-out data for 1-week, 2-week and 4-week 
forecast horizons. 

## How to use this code

1. Download data from the CyanoLakes API using the R scripts provided / use the open api at mobile.cyanolakes.com 
or add your account details for online.cyanolakes.com if you have an account 
2. Save the output file you downloaded (example provided in data directory)
3. Specify your directory settings in settings.py
4. Produce forecasts using main.py, further instructions can be found in the file. 

## Credits
The code was developed by Mark Matthews at CyanoLakes 
during work for the Water Research Commission during 2021 and 2022. 