
#  Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
from sklearn.impute import SimpleImputer

#  Define exponential_growth and err_ranges functions
def exponential_growth(x, a, b, c):
    return a * np.exp(b * x) + c

def err_ranges(x, covariance_matrix):
    sigma = np.sqrt(np.diag(covariance_matrix))
    upper_bound = exponential_growth(x, *popt + 1.96 * sigma)
    lower_bound = exponential_growth(x, *popt - 1.96 * sigma)
    return lower_bound, upper_bound

#  Load World Bank data
data = pd.read_excel('/content/API_19_DS2_en_excel_v2_6300761.xls', skiprows=3)

#  Select relevant columns dynamically and clean up column names
selected_columns = ['Country Name', 'Indicator Name', 'Indicator Code'] + list(map(str, range(1960, 2023)))
data.columns = [str(col[0]) if isinstance(col, tuple) and len(col) > 0 else col for col in data.columns]
data = data[selected_columns]

#  Drop rows with missing values in key columns
data = data.dropna(subset=['Country Name', 'Indicator Name', 'Indicator Code'])

#  Transpose the data for easy manipulation
data = data.set_index(['Country Name', 'Indicator Name', 'Indicator Code']).transpose()
data = data.reset_index()
#  Flatten MultiIndex
data.columns = [' '.join(map(str, col)).strip() for col in data.columns]

#  Flatten the MultiIndex columns
data.columns = [f"{col[1]}_{col[0]}" if isinstance(col, tuple) and not pd.isna(col[0]) else col for col in data.columns]
# Replace 'Unknown' with a unique identifier
data.columns = [f"Unknown_{i}" if col == 'Unknown' else col for i, col in enumerate(data.columns)]
# Remove any remaining characters that are not alphanumeric or underscores
data.columns = ["".join(c for c in col if c.isalnum() or c == '_') for col in data.columns]

#  Perform clustering
X = data.iloc[:, 3:].values  # Select columns from 1960 to 2022
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_imputed)

#  Convert the index to integers
data.index = data.index.astype(int)
data.columns = data.columns.str.replace(r'[^a-zA-Z0-9]', '')

# Define indicator name and column name

column_name = 'UnitedKingdomCO2emissionsfromsolidfuelconsumptionktENATMCO2ESFKT'  # Replace this with the specific column you want to visualize


#  Scatter plot for clustering results
plt.scatter(data.index, data[column_name].values, c=data['Cluster'], cmap='viridis')
plt.title('Clustering Results')
plt.xlabel('Year')
plt.ylabel('United Kingdom CO2 Emissions \n From Solid Fuel Consumption(kt)')
plt.savefig('ScatterPlot.png', bbox_inches='tight')
plt.show()

#  Line plot for trend over time
line_column_name = 'ArabWorldCO2emissionsfromliquidfuelconsumptionktENATMCO2ELFKT'
plt.plot(data.index, data[line_column_name].values, label='Arab World CO2 Emissions Trend')
plt.title('Trend Over Time')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions from Liquid Fuel Consumption (kt)')
plt.savefig('LinePlot.png', bbox_inches='tight')

plt.legend()
plt.show()

#  Bar plot for average values across clusters
bar_column_name = 'SpainUrbanpopulationSPURBTOTL'
data.groupby('Cluster')[bar_column_name].mean().plot(kind='bar', color='skyblue')
plt.title('Average Values Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Urban Population (Spain)')
plt.savefig('Plot.png', bbox_inches='tight')

plt.show()

# Box plot for distribution across clusters
box_column_name = 'SpainAgriculturallandsqkmAGLNDAGRIK2'
data.boxplot(column=box_column_name, by='Cluster')
plt.title('Distribution Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Agricultural Land Area (sq km, Spain)')
plt.savefig('BoxPlot.png', bbox_inches='tight')

plt.show()

# Select one country from each cluster for further analysis
countries_for_analysis = data.groupby('Cluster').apply(lambda x: x.iloc[0])



# Perform curve fitting
x = np.arange(len(data.columns[3:]))  # Use an array of integers as x values
y = data.iloc[:, 3:].mean(axis=0).values
nan_mask = np.isnan(y)
y = np.nan_to_num(y)
popt, pcov = curve_fit(exponential_growth, x, y)

# Estimate confidence intervals
lower_bound, upper_bound = err_ranges(x, pcov)

# Curve fitting results plot
plt.plot(data.columns[3:], y, label='Original Data')
plt.plot(data.columns[3:], exponential_growth(x, *popt), label='Best Fitting Function', color='red')
plt.fill_between(data.columns[3:], lower_bound, upper_bound, color='orange', alpha=0.2, label='Confidence Interval')
plt.title('Curve Fitting Results')
plt.xlabel('Year')
plt.ylabel('Mean Indicator Value')
plt.savefig('CurveFitting.png', bbox_inches='tight')
plt.legend()
plt.show()



# GitHub Repository: https://github.com/muhammad-kamran-awan/World-Bank-Dataset-Analysis.git


