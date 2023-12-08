import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from statsmodels.nonparametric.smoothers_lowess import lowess
import geopandas as gpd
import matplotlib
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error
from matplotlib import gridspec
from scipy.optimize import curve_fit
import pickle

# Importing necessary libraries

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
sns.set_style('whitegrid')

df_hist = pd.read_csv('data/XN_hist_data.csv')
df_hist['normalized_dist'] = df_hist['normalized_dist'].astype(float)
print(len(df_hist))
print(len(df_hist['normalized_dist'].dropna()))

# Set font size and style
sns.set(font_scale=1.0, style="white")

# Create figure and axes
fig, ax = plt.subplots(figsize=(4.5, 1.75))

# Plot histogram
sns.histplot(data=df_hist, x='normalized_dist', kde=False, fill=True,
             alpha=.5, edgecolor='black', stat="percent", binwidth=.05, color='#4FA0CA',linewidth=.50)

# Set axis labels and title
ax.set_xlabel('$X_N$', fontsize=6)
ax.set_ylabel('Percent (%)', fontsize=6)

ax.tick_params(axis='both', which='major', labelsize=6, width=0, length=2)
ax.spines['bottom'].set_linewidth(0.5)  # Adjust the line width of the x-axis
ax.spines['left'].set_linewidth(0.5)    # Adjust the line width of the y-axis
sns.despine()

#%%
# Load data
df = pd.read_csv('/Users/jakegearon/CursorProjects/RoRA_NatureSubmit/submission_data_final.xlsx')
# Preprocessing calculations
df['XV_mean'] = df[['XV1', 'XV2', 'XV3']].mean(axis=1)
df['parent_ar_slope_mean'] = df[['parent_ar_slope1', 'parent_ar_slope2', 'parent_ar_slope3']].mean(axis=1)
df['XS/DS'] = df['XV_mean'] / df['parent_ar_slope_mean']
df['spr'] = df['normalized_dist'] / df['parent_ar_slope_mean']

# Calculate the standard error for the mean of XV and parent_ar_slope
df['XV_mean_error'] = df[['XV1', 'XV2', 'XV3']].std(axis=1) / np.sqrt(3)
df['parent_ar_slope_mean_error'] = df[['parent_ar_slope1', 'parent_ar_slope2', 'parent_ar_slope3']].std(axis=1) / np.sqrt(3)

# Calculate gamma and its log-transform
df['log_gamma'] = np.log(df['XV_mean']) - np.log(df['parent_ar_slope_mean'])
df['gamma'] = np.exp(df['log_gamma'])

# Calculate the standard error of the log-transformed means
df['log_XV_mean_SE'] = df['XV_mean_error'] / df['XV_mean']
df['log_parent_ar_slope_mean_SE'] = df['parent_ar_slope_mean_error'] / df['parent_ar_slope_mean']

# Calculate the absolute error for gamma in the log domain and adjust for the back-transform
df['log_gamma_error'] = np.sqrt(df['log_XV_mean_SE']**2 + df['log_parent_ar_slope_mean_SE']**2)
df['gamma_error_upper'] = np.exp(df['log_gamma'] + df['log_gamma_error']) - df['gamma']

# Calculate the lower error and ensure it doesn't go below the minimum allowed value after back-transform
df['gamma_error_lower'] = np.maximum(df['gamma'] - np.exp(df['log_gamma'] - df['log_gamma_error']), 0.1)

# Calculate the relative errors adjusted for the log scale plot
yerr_upper_relative = df['gamma_error_upper'] / df['XS/DS']
yerr_lower_relative = df['XS/DS'] - np.maximum(df['XS/DS'] - df['gamma_error_lower'], 0.1)

# Load XGBoost model and make predictions
xgb_reg = xgb.XGBRegressor()
xgb_reg.load_model("/Users/jakegearon/CursorProjects/based/based/models/based_us_early_stopping_combat_overfitting.ubj")

# Apply the power-law correction to the discharge values
df['corrected_discharge'] = inverse_power_law(df['discharge_max'], *params)
guesswork_full = df[['width', 'parent_ar_slope_mean', 'corrected_discharge']].astype(float)
guesswork_full.columns = ['width', 'slope', 'discharge']
# Use the corrected discharge, width, and parent_ar_slope_mean to predict the depth
df['XGB_depth'] = xgb_reg.predict(guesswork_full)
df['XGB_depth'] = np.where(df['XGB_depth'] < 0, 0.1, df['XGB_depth'])
# More data transformations
df['alluvial_ridge_height1'] = df['alluvial_ridge_elevation1'] - df['floodplain_elevation1']
df['alluvial_ridge_height2'] = df['alluvial_ridge_elevation2'] - df['floodplain_elevation2']
df['alluvial_ridge_height3'] = df['alluvial_ridge_elevation3'] - df['floodplain_elevation3']

# Superelevation calculations
df['A'] = df[['alluvial_ridge_elevation1', 'alluvial_ridge_elevation2', 'alluvial_ridge_elevation3']].sub(df[['water_elevation1', 'water_elevation2', 'water_elevation3']].values).mean(axis=1)
df['B'] = df['XGB_depth']
# Assuming you have a DataFrame named 'df'

# Calculate 'A1' and 'B1' for the first set of columns (2)
df['A1'] = df['alluvial_ridge_elevation1'] - df['water_elevation1']
df['B1'] = df['XGB_depth']

# Calculate 'A2' and 'B2' for the second set of columns (3)
df['A2'] = df['alluvial_ridge_elevation2'] - df['water_elevation2']
df['B2'] = df['XGB_depth']

# Calculate 'A3' and 'B3' for the third set of columns (4)
df['A3'] = df['alluvial_ridge_elevation3'] - df['water_elevation3']
df['B3'] = df['XGB_depth']

# Find the mean of 'A' values (A1, A2, and A3) and 'B' values (B1, B2, and B3)
df['A'] = df[['A1', 'A2', 'A3']].mean(axis=1)
df['B'] = df[['B1', 'B2', 'B3']].mean(axis=1)

# Calculate A/B
df['A/B'] = abs(df['A']) / abs(df['B'])


# Masks for each category
mask1 = df['method_used2'] == 1
mask2 = df['method_used2'] == 2
mask3 = df['method_used2'] == 3

df.loc[mask1, 'se_max1'] = df['alluvial_ridge_height1'] / df['XGB_depth']
df.loc[mask1, 'se_max2'] = df['alluvial_ridge_height2'] / df['XGB_depth']
df.loc[mask1, 'se_max3'] = df['alluvial_ridge_height3'] / df['XGB_depth']


df.loc[mask2, 'se_max1'] = df['alluvial_ridge_height1'] / (df['alluvial_ridge_elevation1'] - (df['water_elevation1']))
df.loc[mask2, 'se_max2'] = df['alluvial_ridge_height2'] / (df['alluvial_ridge_elevation2'] - (df['water_elevation2']))
df.loc[mask2, 'se_max3'] = df['alluvial_ridge_height3'] / (df['alluvial_ridge_elevation3'] - (df['water_elevation3']))

# When A/B > 4
df.loc[mask3, 'se_max1'] = df['alluvial_ridge_height1'] / (df['alluvial_ridge_elevation1'] - (df['water_elevation1']-df['XGB_depth']))  # Fill this in with the relevant formula
df.loc[mask3, 'se_max2'] = df['alluvial_ridge_height2'] / (df['alluvial_ridge_elevation2'] - (df['water_elevation2']-df['XGB_depth']))  # Fill this in with the relevant formula
df.loc[mask3, 'se_max3'] = df['alluvial_ridge_height3'] / (df['alluvial_ridge_elevation3'] - (df['water_elevation3']-df['XGB_depth']))  # Fill this in with the relevant formula


df['se_std'] = df[['se_max1', 'se_max2', 'se_max3']].std(axis=1)
df['superelevation_ov_mean'] = df[['se_max1', 'se_max2', 'se_max3']].mean(axis=1)
df['superelevation_sem'] = df[['se_max1', 'se_max2', 'se_max3']].sem(axis=1) / 3

# Additional processing
# df['Geomorphology'] = df['Geomorphology'].replace({'Mountain Front': 'Alluvial Plain'})

df['prod'] = df['superelevation_ov_mean'] * df['XS/DS']
df['AB_flag'] = np.abs(df['A']) > np.abs(df['B'])
#%%

# Setting visualization parameters

y_test = pd.read_csv("data/based_us_ytest_data_sans_trampush.csv", usecols=["depth"]).astype(float)
X_test = pd.read_csv("data/t/based_us_Xtest_data_sans_trampush.csv")
trampush_csv = pd.read_csv("data/trampyupdatercsv.csv").dropna(subset=["RA_dis_m3_pmx_2", "Qbf [m3/s]"])

# Loading test data and data to be processed

trampush_csv = trampush_csv.query("Location != 'Ditch Run near Hancock'")
trampush_csv = trampush_csv.query("Location != 'Potomac River Tribuatary near Hancock. Md.'")
trampush_csv = trampush_csv.query("Location != 'Clark Fork tributary near Drummond'")
trampush_csv = trampush_csv.query("Location != 'Richards Creek near Libby MT'")
trampush_csv = trampush_csv.query("Location != 'Ohio Brush Creek near West Union'")
trampush_csv = trampush_csv[trampush_csv["Location"] != "Floyd's Fork at Fisherville"]

# Removing misaligned data

xgb_reg = xgb.XGBRegressor()
xgb_reg.load_model("/Users/jakegearon/CursorProjects/based/based/models/based_us_sans_trampush_early_stopping_combat_overfitting.ubj")

# Loading the XGBoost model

def power_law(x, a, b):
    return a * (x ** b)

def inverse_power_law(y, a, b):
    return (y / a) ** (1 / b)

# Defining power law and its inverse

x_data = trampush_csv['Qbf [m3/s]'].values
y_data = trampush_csv['RA_dis_m3_pmx_2'].values
params, covariance = curve_fit(power_law, x_data, y_data)
min_val = min(trampush_csv['Qbf [m3/s]'].min(), trampush_csv['RA_dis_m3_pmx_2'].min())
max_val = max(trampush_csv['Qbf [m3/s]'].max(), trampush_csv['RA_dis_m3_pmx_2'].max())
x_fit = np.linspace(min_val, max_val, 100)
y_fit = power_law(x_fit, *params)

# Fitting power law to the data

with open('/Users/jakegearon/CursorProjects/based/data/inverted_discharge_params.pickle', 'wb') as f:
    pickle.dump(params, f)

# Saving parameters to a file

trampush_csv['corrected_discharge'] = inverse_power_law(trampush_csv['RA_dis_m3_pmx_2'], *params)

# Calculating the corrected discharge values

guessworkRA = trampush_csv[['Wbf [m]', 'S [-]', 'RA_dis_m3_pmx_2']].astype(float)
guessworkRA.columns = ['width', 'slope', 'discharge']
trampush_csv['XGB_depth_RA'] = xgb_reg.predict(guessworkRA)

# Predicting the depth values for modeled discharge values

depth_predictions = pd.Series(xgb_reg.predict((X_test[['width', 'slope', 'discharge']]))).astype(float)

# Predicting the depth values for the test data

guessworkM = trampush_csv[['Wbf [m]', 'S [-]', 'Qbf [m3/s]']].astype(float)
guessworkM.columns = ['width', 'slope', 'discharge']
trampush_csv['XGB_depth_M'] = xgb_reg.predict(guessworkM)

# Predicting the depth values for the manually recorded discharge values

guessworkRA_corr = trampush_csv[['Wbf [m]', 'S [-]', 'corrected_discharge']].astype(float)
guessworkRA_corr.columns = ['width', 'slope', 'discharge']
trampush_csv['XGB_depth_RA_corr'] = xgb_reg.predict(guessworkRA_corr)

# Predicting the depth values for the corrected discharge values

#%%
import matplotlib.ticker as ticker

# Custom formatter function
def custom_formatter(x, pos):
    # Check if the value is less than 1 and format it with an additional decimal place
    if x < 1:
        return '{:.1f}'.format(x)  # 1 decimal places for values less than 1
    elif x <= 0.1:
        return '{:.2f}'.format(x) # 2 decimal places for values less than .1
    elif x == 0:
        return '{:.0f}'.format(x)
    else:
        return '{:.0f}'.format(x)  # 0 decimal places for all other values
    
# Create a figure
fig_width, fig_height = 7, 3.5  # Adjusted width to accommodate the new column
fig = plt.figure(figsize=(fig_width, fig_height))

# Divide the figure into 7 rows and 3 columns
gs1 = gridspec.GridSpec(7, 3, width_ratios=[.5, .55, 1.95], height_ratios=[2.5, .5, .5, .5, .5, .5, .5], wspace=.10, hspace=.7)  # Updated width ratios

ax_new_hist = fig.add_subplot(gs1[0, 2:3])  
sns.histplot(data=df_hist, x='normalized_dist', kde=False, fill=True,
             alpha=1, edgecolor='black', stat="percent", binwidth=.05, color='#56b4e9', linewidth=.65, ax=ax_new_hist) #'#4FA0CA'
#ax_new_hist.grid(axis='y', which='major', color='gray', linestyle='-', linewidth=.5, visible=False)
ax_new_hist.grid(False)
ax_new_hist.set_ylim(0, 45)
# Set axis labels and title for the new histogram
ax_new_hist.set_xlabel('$X_N$', fontsize=6, labelpad=-3)
ax_new_hist.set_ylabel('% Occurrence', fontsize=6, labelpad=-.5)
ax_new_hist.tick_params(axis='both', which='major', labelsize=6, width=0, length=2)
sns.despine(ax=ax_new_hist)

# Scatter plots on the left
ax0 = fig.add_subplot(gs1[1:4, 2])

ax0.errorbar(boot_df['sp'], boot_df['superelevation'], yerr=boot_df['uncertainty'], capsize=3.5, capthick=0.5, ls='none', lw=.5, color='black', markeredgecolor='black', markersize=8, zorder=1)
plot_scatter_with_lowess(ax0, df, 'sp', 'superelevation_ov_mean', color_data='Geomorphology', palette=palette)
ax0.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
ax0.set_xlabel('')
ax0.set_ylabel('')
ax0.set_xscale('log')


ax1 = fig.add_subplot(gs1[4:7, 2])
ax1.errorbar(df['sp'], df['XS/DS'], yerr=[yerr_lower_relative, yerr_upper_relative], capsize=3.5, capthick=0.5, ls='none', lw=.5, color='black', markeredgecolor='black', markersize=8, zorder=1)
plot_scatter_with_lowess(ax1, df, 'sp', 'XS/DS', color_data='Geomorphology', palette=palette)
ax1.set_xlabel(r'proximal and steep $\longleftarrow SPR^{\star} \longrightarrow$ distal and gentle', fontsize=fontsize, labelpad=1)
ax1.set_xscale('log')
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))



# Boxplots in the middle (unchanged)
ax3 = fig.add_subplot(gs1[1:4, 1])
plot_boxplot(ax3, 'Geomorphology', 'superelevation_ov_mean', ['Fan', 'Alluvial Plain', 'Delta'])
ax3.set_xlabel('')
ax3.set_ylabel('')

ax4 = fig.add_subplot(gs1[4:7, 1])
plot_boxplot(ax4, 'Geomorphology', 'XS/DS', ['Fan', 'Alluvial Plain', 'Delta'])
ax4.set_xlabel('')
ax4.set_ylabel('')
ax4.set_xlabel('Geomorphology', fontsize=fontsize)
# Histograms on the right (unchanged)
ax6 = fig.add_subplot(gs1[1:4, 0])
sns.histplot(data=df, y='superelevation', bins=10, edgecolor='k', color='white', alpha=1, stat="probability", ax=ax6, zorder=3, linewidth=.5)
ax6.set_ylabel(r'$\beta$', rotation=0, fontsize=fontsize)
ax6.set_yticks([0, 1, 2, 3, 4, 5])
ax6.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax6.grid(False)
ax6.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))



ax6.set_xlabel('')
ax6.axhspan(.5, 1.1, color='#BDBDBD', alpha=.2, fill=True, zorder=1)
ax6.set_ylim(0, 5.5)
median_superelevation = df['superelevation_ov_mean'].median()
ax6.axhline(y=median_superelevation, color='black', linestyle='-.', linewidth=.75, zorder=4)

ax7 = fig.add_subplot(gs1[4:7, 0])
ax7.grid(False)
sns.histplot(data=df, y='XS/DS', bins=10, edgecolor='k', color='white', alpha=1, stat="probability", ax=ax7, zorder=3, linewidth=.5)
ax7.set_ylabel(r'$\gamma$', rotation=0, fontsize=fontsize)
ax7.set_xlabel('Probability', fontsize=fontsize)
# Plotting the right histogram with shaded area
ax7.axhspan(3, 10, color='#BDBDBD', alpha=0.2, zorder=1)
ax7.set_ylim(-.5, 35)
median_XS_DS = df['XS/DS'].median()
ax7.axhline(y=median_XS_DS, color="black", linestyle='-.', linewidth=.75, zorder=4)
ax7.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax7.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))


all_axes = [ax_new_hist, ax0, ax1, ax3, ax4, ax6, ax7]
for ax in all_axes:
    for spine in ax.spines.values():
        spine.set_linewidth(.5)
    if ax == ax_new_hist:
        ax.tick_params(axis='both', labelsize=5, pad=-.2)
    else:
        ax.tick_params(axis='both', labelsize=5, pad=-3.5)



plt.margins(0, 0)
plt.savefig('FIGURE2.pdf')
plt.savefig('FIGURE2.png', dpi=300)
plt.show()


#%%


