import pmdarima as pm
from pmdarima.datasets import load_wineind

y = load_wineind()
train, test = y[:125], y[125:]

# Fit an ARIMA
arima = pm.ARIMA(order=(1, 1, 2), seasonal_order=(0, 1, 1, 12))
arima.fit(y)