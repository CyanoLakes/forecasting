## Simple univariate forecasting models

This code provides a comparison of simple baseline univariate forecasting models and the moving average seasonal error adjusted (MASEA) model developed by CyanoLakes used to forecast cyanobacteria blooms and chlorophyll-a concentrations in lakes. Forecasts are made using satellite remotely sensed data from the CyanoLakes Mobile and Web Applications (data is available via an API). Forecasts are produced using 1 year of hold-out data for 1-week, 2-week and 4-week forecast horizons. 

Further details can be found in the article published in Inland Waters accepted on 2 November 2022. [Near-term forecasting of cyanobacterial and harmful algal blooms in lakes using simple univariate methods with satellite remote sensing data](https://github.com/CyanoLakes/forecasting/files/9947016/Accepted_version.pdf).



## How to use this code
1. Use the chlorophyll-a time-series data provided in the data directory. If you need more data, contact the author. 
2. Specify your directory settings in settings.py
3. Produce forecasts using main.py, further instructions can be found in the comments.  

## Credits
The code was developed by Mark Matthews at CyanoLakes during work for the Water Research Commission during 2021 and 2022. This work was supported by the Water Research Commission under Grant C2019_2020-00198. This code is provided without warranty for non-commercial research purposes only. 

