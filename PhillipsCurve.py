import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
unemployment_data = pd.read_csv('unemployment_rate.csv')
cpi_data = pd.read_csv('CPI.csv')

# Convert DATE columns to datetime
unemployment_data['DATE'] = pd.to_datetime(unemployment_data['DATE'])
cpi_data['DATE'] = pd.to_datetime(cpi_data['DATE'])

# Merge datasets on DATE
merged_data = pd.merge(unemployment_data, cpi_data, on='DATE', how='inner')

# Calculate inflation rate as the percentage change in CPI
merged_data['InflationRate'] = merged_data['CPILFESL'].pct_change() * 100

# Drop the rows with NaN values that result from pct_change()
merged_data = merged_data.dropna()

# Plotting the Phillips Curve
plt.figure(figsize=(10, 6))
plt.scatter(merged_data['UNRATE'], merged_data['InflationRate'])
plt.title('Phillips Curve')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Inflation Rate (%)')
plt.grid(True)
plt.show()
