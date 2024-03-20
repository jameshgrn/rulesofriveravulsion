import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import geopandas as gpd
import pandas as pd
from scipy.stats import linregress
import matplotlib
import pickle

palette = ['#90EE90', '#F5F5DC', '#ADD8E6']
sns.set_context('paper', font_scale=1.0)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 5

# Load your data
df_val = pd.read_excel('data/manuscript_data/Supplementary Table 3.xlsx')

df_grdi_basins = gpd.read_file('data/extended_data_figure_data/level_7_avulsion_basins_grdi.geojson')

# Set the style for the plot
sns.set(style="white")

# Create two subplots
fig, axs = plt.subplots(1, 2, figsize=(4, 2.0))

def inverse_power_law(y, a, b):
    return (y / a) ** (1 / b)

# Function to calculate r-squared and p-value
def calc_stats(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2, p_value

# Plot for Alluvial Ridge Height
r2_height, p_height = calc_stats(df_val['FABDEM'], df_val['IS2'])
axs[0].scatter(df_val['FABDEM'], df_val['IS2'], alpha=0.8, c='blue', edgecolor='k', s=20, zorder=2)
axs[0].plot([0, 15], [0, 15], linestyle='--', color='gray', zorder=1)
slope_height, intercept_height = np.polyfit(df_val['FABDEM'], df_val['IS2'], 1)
axs[0].plot(df_val['FABDEM'], intercept_height + slope_height * df_val['FABDEM'], 'r', label='OLS', lw=.5, ls=':')
axs[0].set_xlabel('FABDEM $H_{AR}$', fontsize=7)
axs[0].set_ylabel('IceSat-2 $H_{AR}$', fontsize=7)
axs[0].set_xticks([0, 5, 10, 15])
axs[0].set_yticks([0, 5, 10, 15])
axs[0].set_xlim(0, 15)
axs[0].set_ylim(0, 15)
axs[0].legend([f'R²={r2_height:.2f}, p={p_height:.2e}'], loc='upper left', frameon=True, fontsize=6)

# Plot for Alluvial Ridge Slope
r2_slope, p_slope = calc_stats(df_val['FABDEM_SLOPE'], df_val['IS2_SLOPE'])
axs[1].scatter(df_val['FABDEM_SLOPE'], df_val['IS2_SLOPE'], alpha=0.8, c='green', edgecolor='k', s=20, zorder=2)
axs[1].plot([0, .009], [0, .009], linestyle='--', color='gray', zorder=1)
slope_slope, intercept_slope = np.polyfit(df_val['FABDEM_SLOPE'], df_val['IS2_SLOPE'], 1)
axs[1].plot(df_val['FABDEM_SLOPE'], intercept_slope + slope_slope * df_val['FABDEM_SLOPE'], 'r', label='OLS', lw=.5, ls=':')
axs[1].set_xlabel('FABDEM $S_{AR}$', fontsize=7)
axs[1].set_ylabel('IceSat-2 $S_{AR}$', fontsize=7)
axs[1].set_xticks([0, .002, .004, .006, .008])
axs[1].set_yticks([0, .002, .004, .006, .008])
axs[1].set_xlim(0, .009)
axs[1].set_ylim(0, .009)
axs[1].legend([f'R²={r2_slope:.2f}, p={p_slope:.2e}'], loc='upper left', frameon=True, fontsize=6)

# Adjust layout
plt.subplots_adjust(wspace=0.45)
for ax in axs:
    ax.grid(True, linestyle='--', color='grey', lw=0.5, zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=6, pad=-1.5)
    ax.set_aspect('equal', 'box')
    
axs[0].text(.038, 1.06, 'A', transform=axs[0].transAxes, 
         fontsize=6, fontweight='bold', va='top', ha='right')

# Label for the scatter plot
axs[1].text(.028, 1.06, 'B', transform=axs[1].transAxes, 
         fontsize=6, fontweight='bold', va='top', ha='right')

# Save the figure
fig.savefig('figure_plotting/figures/extended_data_figs/extended_data_figure_2.png', dpi=300, bbox_inches='tight')
plt.close(fig)

data = [
    ["Indonesia", 66.033907],
    ["Argentina", 66.363962],
    ["Cyprus", 48.437801],
    ["India", 68.516289],
    ["People's Republic of China", 42.793618],
    ["France", 55.236897],
    ["South Korea", 43.272558],
    ["South Africa", 68.943652],
    ["Lithuania", 59.382433],
    ["Brazil", 57.594965],
    ["Czech Republic", 49.566262],
    ["Germany", 48.471918],
    ["Estonia", 60.122871],
    ["Latvia", 61.212536],
    ["Sweden", 58.857821],
    ["Finland", 59.860711],
    ["Luxembourg", 43.891102],
    ["Belgium", 37.563061],
    ["Turkey", 57.619970],
    ["Spain", 53.400327],
    ["Denmark", 55.345361],
    ["Romania", 56.866002],
    ["Hungary", 53.375399],
    ["Slovakia", 55.908528],
    ["Poland", 54.110025],
    ["Ireland", 59.063961],
    ["United Kingdom", 52.737709],
    ["Greece", 53.788649],
    ["Austria", 53.506250],
    ["Italy", 42.543722],
    ["Croatia", 53.481072],
    ["Slovenia", 53.256296],
    ["Saudi Arabia", 55.757394],
    ["Bulgaria", 55.662894],
    ["Portugal", 46.100522],
    ["United States of America", 54.884098],
    ["Canada", 57.569389],
    ["Mexico", 65.726648],
    ["Australia", 55.170738],
    ["Japan", 49.164411],
    ["Malta", 30.840760],
    ["Netherlands", 40.603666]
]

eu_countries = {"Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", 
               "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", 
               "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", 
               "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"}

non_eu_dict = {country[0]: country[1] for country in data if country[0] not in eu_countries}

eu_mean = np.mean([48.437801, 55.236897, 59.382433, 49.566262, 48.471918, 60.122871,
         61.212536, 58.857821, 59.860711, 43.891102, 37.563061, 53.400327,
         55.345361, 56.866002, 53.375399, 55.908528, 54.110025, 59.063961,
        53.788649, 53.506250, 42.543722, 53.481072, 53.256296, 55.662894
        ,46.100522, 30.840760, 40.603666])
eu_mean
non_eu_dict['EU'] = eu_mean
final_dict_w_eu = non_eu_dict
final_dict_df = pd.DataFrame(list(final_dict_w_eu.items()), columns=['Country', 'GRDI_mean'])
final_dict_df = gpd.read_file('data/extended_data_figure_data/g20_grdi.geojson')
fig, ax = plt.subplots(figsize=(7, 4))  # Adjust the figure size as needed
# Your histplot code
ax.fill_between([df_grdi_basins['GRDI_mean'].mean(), final_dict_df['GRDI_mean'].mean()], 0, 30,
                  hatch='/', edgecolor='gray', alpha=0.1, zorder=0)  # Set xmin and xmax
sns.histplot(data=final_dict_df, x='GRDI_mean', kde=False, fill=True,
             alpha=.5, edgecolor='black', stat="percent", color='red',linewidth=.50, binwidth=4, zorder=2, ax=ax)
sns.histplot(data=df_grdi_basins, x='GRDI_mean', kde=False, fill=True,
             alpha=.5, edgecolor='black', stat="percent", color='#4FA0CA',linewidth=.50, zorder=1, ax=ax)

ax.axvline(df_grdi_basins['GRDI_mean'].mean(), c='black', ls='--', lw=2, label='Avulsion Dataset Mean')
ax.axvline(final_dict_df['GRDI_mean'].mean(), c='black', ls=':', lw=2, label='G20 Mean')
ax.set_ylim(0, 26)
ax.legend()
ax.set_xlabel('Relative Deprivation Index')
plt.tight_layout()
fig.savefig('figure_plotting/figures/extended_data_figs/extended_data_figure_5.png', dpi=300)
plt.close(fig)


def inverse_power_law(y, a, b):
    return (y / a) ** (1 / b)

with open('data/BASED_model/inverse_power_law_params.pickle', 'rb') as f:
    params = pickle.load(f)

df = pd.read_csv('data/figure2_data/fig2_data.csv')
# Apply the power-law correction to the discharge values
df['corrected_discharge_mean'] = inverse_power_law(df['discharge_mean_cms'], *params)
df['corrected_discharge_min'] = inverse_power_law(df['discharge_min_cms'], *params)
df['newdis'] = (df['corrected_discharge'] - df['corrected_discharge_min'])/df['corrected_discharge_mean']
tdf = df.copy().dropna(subset=['newdis', 'beta'])
# Assuming you have already defined 'newdis' and loaded other columns from your DataFrame

# Create a scatter plot
fig, ax = plt.subplots(figsize=(3, 3))  # Adjust the figure size as needed

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(tdf['newdis'], tdf['beta'])

# # Plot the scatter plot
sns.scatterplot(x=tdf['newdis'], y=tdf['beta'], hue=tdf['geomorphology'], s=60,
                palette=palette, hue_order=['Fan', 'Alluvial Plain', 'Delta'], edgecolor='k', ax=ax)

# Plot the linear regression line
x_range = np.linspace(min(tdf['newdis']), max(tdf['newdis']), 100)
y_pred = slope * x_range + intercept
ax.plot(x_range, y_pred, color='black', linestyle='--', label=f'OLS (R²={r_value**2:.2f}, p={p_value:.2f})')

# Add a legend with R² and p-value
ax.legend()

# Customize plot labels, titles, and aesthetics as needed
ax.set_xlabel(r'$Q^{\star}$')
ax.set_ylabel(r'$\beta$', rotation=0)
#plt.title('Scatter Plot with Linear Regression Fit')
ax.grid(True)
plt.tight_layout()
fig.savefig('figure_plotting/figures/extended_data_figs/extended_data_figure_4.png', dpi=300)
plt.close(fig)


#extended data figure 1 base
gdf = gpd.read_feather('data/avulsion_data/TURK_002_1991/TURK_002_1991_UID_3_data_water_1.feather')
gdf = gdf[(gdf['atl03_cnf'] >= 0) & (gdf['atl08_class'] == 1)]
gdf["along_track"] = gdf["along_track"] - gdf["along_track"].min()

fig, ax = plt.subplots()
sns.scatterplot(data=gdf, x="along_track", y="height", s=2, color="black", alpha=0.5, ax=ax)
plt.show()





