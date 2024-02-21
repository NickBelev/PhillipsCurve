import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Load the unemployment and CPI data
unemployment_data = pd.read_csv('unemployment_rate.csv')
cpi_data = pd.read_csv('CPI.csv')

# Make sure dates are in datetime format
unemployment_data['DATE'] = pd.to_datetime(unemployment_data['DATE'])
cpi_data['DATE'] = pd.to_datetime(cpi_data['DATE'])

# Grab the year from each DATE for grouping
unemployment_data['YEAR'] = unemployment_data['DATE'].dt.year
cpi_data['YEAR'] = cpi_data['DATE'].dt.year

# Get the yearly avg unemployment rate
annual_unemployment = unemployment_data.groupby('YEAR')['UNRATE'].mean().reset_index()

# Average CPI per year, then find the inflation rate
cpi_data['CPILFESL'] = cpi_data.groupby('YEAR')['CPILFESL'].transform('mean')
cpi_data = cpi_data.drop_duplicates('YEAR')
cpi_data['InflationRate'] = cpi_data['CPILFESL'].pct_change() * 100

annual_inflation = cpi_data[['YEAR', 'InflationRate']].dropna()

# Merge the two datasets on year
merged_annual_data = pd.merge(annual_unemployment, annual_inflation, on='YEAR', how='inner')

# Hyperbolic function for the curve fitting
def hyperbolic(x, a, b):
    return a / x + b

# Fit the data to our hyperbolic function
params, _ = curve_fit(hyperbolic, merged_annual_data['UNRATE'], merged_annual_data['InflationRate'])

# Prep for plotting
x_values = np.linspace(merged_annual_data['UNRATE'].min(), merged_annual_data['UNRATE'].max(), 100)
y_values = hyperbolic(x_values, *params)

# Manually calculate R squared for the fit
predicted_inflation = hyperbolic(merged_annual_data['UNRATE'], *params)
ss_res = np.sum((merged_annual_data['InflationRate'] - predicted_inflation) ** 2)
ss_tot = np.sum((merged_annual_data['InflationRate'] - np.mean(merged_annual_data['InflationRate'])) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Draw the plot
plt.figure(figsize=(10, 6))
plt.scatter(merged_annual_data['UNRATE'], merged_annual_data['InflationRate'], label='Data')
plt.plot(x_values, y_values, 'r-', label=f'Fit: $R^2$ = {r_squared:.3f}')
plt.title('Phillips Curve: Unemployment vs Inflation')
plt.xlabel('Annual Unemployment Rate (%)')
plt.ylabel('Annual Inflation Rate (%)')
plt.legend()
plt.grid(True)
plt.show()
