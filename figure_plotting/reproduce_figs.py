#%%
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

def power_law(x, a, b):
    return a * (x ** b)

def inverse_power_law(y, a, b):
    return (y / a) ** (1 / b)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
sns.set_style('whitegrid')

df_hist = pd.read_excel('data/manuscript_data/Supplementary Table 1.xlsx')
df_hist['normalized_distance'] = df_hist['normalized_distance'].astype(float)


# Set font size and style
sns.set(font_scale=1.0, style="white")

# Create figure and axes
fig, ax = plt.subplots(figsize=(4.5, 1.75))

# Plot histogram
sns.histplot(data=df_hist, x='normalized_distance', kde=False, fill=True,
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
df = pd.read_excel('/Users/jakegearon/CursorProjects/RoRA_NatureSubmit/submission_data_final.xlsx')
# Preprocessing calculations
df['sar_mean'] = df[['sar1', 'sar1', 'sar3']].mean(axis=1)
df['sm_mean'] = df[['sm1', 'sm2', 'sm3']].mean(axis=1)
df['gamma'] = df['sar_mean'] / df['sm_mean']
df['spr'] = df['normalized_distance'] / df['sm_mean']

# Calculate the standard error for the mean of XV and parent_ar_slope
df['sar_mean_error'] = df[['sar1', 'sar2', 'sar3']].std(axis=1) / np.sqrt(3)
df['sm_mean_error'] = df[['sm1', 'sm2', 'sm3']].std(axis=1) / np.sqrt(3)

# Calculate gamma and its log-transform
df['log_gamma'] = np.log(df['sar_mean']) - np.log(df['sm_mean'])

# Calculate the standard error of the log-transformed means
df['log_sar_mean_SE'] = df['sar_mean_error'] / df['sar_mean']
df['log_sm_mean_SE'] = df['sm_mean_error'] / df['sm_mean']

# Calculate the absolute error for gamma in the log domain and adjust for the back-transform
df['log_gamma_error'] = np.sqrt(df['log_sar_mean_SE']**2 + df['log_sm_mean_SE']**2)
df['gamma_error_upper'] = np.exp(df['log_gamma'] + df['log_gamma_error']) - df['gamma']

# Calculate the lower error and ensure it doesn't go below the minimum allowed value after back-transform
df['gamma_error_lower'] = np.maximum(df['gamma'] - np.exp(df['log_gamma'] - df['log_gamma_error']), 0.1)

# Calculate the relative errors adjusted for the log scale plot
df['yerr_upper_relative'] = df['gamma_error_upper'] / df['gamma']
df['yerr_lower_relative'] = df['gamma'] - np.maximum(df['gamma'] - df['gamma_error_lower'], 0.1)

# Load XGBoost model and make predictions
xgb_reg = xgb.XGBRegressor()
xgb_reg.load_model("data/BASED_model/models/based_us_early_stopping_combat_overfitting.ubj")

with open('data/BASED_model/inverse_power_law_params.pickle', 'rb') as f:
    params = pickle.load(f)

# Apply the power-law correction to the discharge values
df['corrected_discharge'] = inverse_power_law(df['discharge_max_cms'], *params)
guesswork_full = df[['width_m', 'sm_mean', 'corrected_discharge']].astype(float)
guesswork_full.columns = ['width', 'slope', 'discharge']
# Use the corrected discharge, width, and parent_ar_slope_mean to predict the depth
df['xgb_depth'] = xgb_reg.predict(guesswork_full)
df['xgb_depth'] = np.where(df['xgb_depth'] < 0, 0.1, df['xgb_depth'])
# More data transformations
df['har1_m'] = df['are1_m'] - df['fpe1_m']
df['har2_m'] = df['are2_m'] - df['fpe2_m']
df['har3_m'] = df['are3_m'] - df['fpe3_m']

# Superelevation calculations
df['a'] = df[['are1_m', 'are2_m', 'are3_m']].sub(df[['wse1_m', 'wse2_m', 'wse3_m']].values).mean(axis=1)
df['b'] = df['xgb_depth']
# Assuming you have a DataFrame named 'df'

# Calculate 'A1' and 'B1' for the first set of columns (2)
df['a1'] = df['are1_m'] - df['wse1_m']
df['b1'] = df['xgb_depth']

# Calculate 'A2' and 'B2' for the second set of columns (3)
df['a2'] = df['are2_m'] - df['wse2_m']
df['b2'] = df['xgb_depth']

# Calculate 'A3' and 'B3' for the third set of columns (4)
df['a3'] = df['are3_m'] - df['wse3_m']
df['b3'] = df['xgb_depth']

# Find the mean of 'A' values (A1, A2, and A3) and 'B' values (B1, B2, and B3)
df['a'] = df[['a1', 'a2', 'a3']].mean(axis=1)
df['b'] = df[['b1', 'b2', 'b3']].mean(axis=1)

# Calculate A/B
df['a_over_b'] = abs(df['a']) / abs(df['b'])


# Masks for each category
mask1 = df['method_used'] == 1
mask2 = df['method_used'] == 2
mask3 = df['method_used'] == 3

df.loc[mask1, 'se_max1'] = df['har1_m'] / df['xgb_depth']
df.loc[mask1, 'se_max2'] = df['har2_m'] / df['xgb_depth']
df.loc[mask1, 'se_max3'] = df['har3_m'] / df['xgb_depth']


df.loc[mask2, 'se_max1'] = df['har1_m'] / (df['are1_m'] - (df['wse1_m']))
df.loc[mask2, 'se_max2'] = df['har2_m'] / (df['are2_m'] - (df['wse2_m']))
df.loc[mask2, 'se_max3'] = df['har3_m'] / (df['are3_m'] - (df['wse3_m']))

# When A/B > 4
df.loc[mask3, 'se_max1'] = df['har1_m'] / (df['are1_m'] - (df['wse1_m']-df['xgb_depth']))  # Fill this in with the relevant formula
df.loc[mask3, 'se_max2'] = df['har2_m'] / (df['are2_m'] - (df['wse2_m']-df['xgb_depth']))  # Fill this in with the relevant formula
df.loc[mask3, 'se_max3'] = df['har3_m'] / (df['are3_m'] - (df['wse3_m']-df['xgb_depth']))  # Fill this in with the relevant formula


df['se_std'] = df[['se_max1', 'se_max2', 'se_max3']].std(axis=1)
df['beta'] = df[['se_max1', 'se_max2', 'se_max3']].mean(axis=1)
df['beta_sem'] = df[['se_max1', 'se_max2', 'se_max3']].sem(axis=1) / 3

# Additional processing
# df['Geomorphology'] = df['Geomorphology'].replace({'Mountain Front': 'Alluvial Plain'})

df['lambda'] = df['beta'] * df['gamma']
df['ab_flag'] = np.abs(df['a']) > np.abs(df['b'])
#%%

# Setting visualization parameters

ytest = pd.read_csv("data/BASED_model/train_test/based_us_ytest_data_sans_trampush.csv", usecols=["depth"]).astype(float)
Xtest = pd.read_csv("data/BASED_model/train_test/based_us_Xtest_data_sans_trampush.csv")
trampush_csv = pd.read_csv("data/BASED_model/TrampushDataClean.csv").dropna(subset=["RA_dis_m3_pmx_2", "Qbf [m3/s]"])

# Loading test data and data to be processed

trampush_csv = trampush_csv.query("Location != 'Ditch Run near Hancock'")
trampush_csv = trampush_csv.query("Location != 'Potomac River Tribuatary near Hancock. Md.'")
trampush_csv = trampush_csv.query("Location != 'Clark Fork tributary near Drummond'")
trampush_csv = trampush_csv.query("Location != 'Richards Creek near Libby MT'")
trampush_csv = trampush_csv.query("Location != 'Ohio Brush Creek near West Union'")
trampush_csv = trampush_csv[trampush_csv["Location"] != "Floyd's Fork at Fisherville"]

# Removing misaligned data

xgb_reg = xgb.XGBRegressor()
xgb_reg.load_model("data/BASED_model/models/based_us_sans_trampush_early_stopping_combat_overfitting.ubj")


# Defining power law and its inverse

x_data = trampush_csv['Qbf [m3/s]'].values
y_data = trampush_csv['RA_dis_m3_pmx_2'].values
params, covariance = curve_fit(power_law, x_data, y_data)
min_val = min(trampush_csv['Qbf [m3/s]'].min(), trampush_csv['RA_dis_m3_pmx_2'].min())
max_val = max(trampush_csv['Qbf [m3/s]'].max(), trampush_csv['RA_dis_m3_pmx_2'].max())
x_fit = np.linspace(min_val, max_val, 100)
y_fit = power_law(x_fit, *params)

# Fitting power law to the data

with open('data/BASED_model/inverse_power_law_params.pickle', 'rb') as f:
    params = pickle.load(f)

# Saving parameters to a file

trampush_csv['corrected_discharge'] = inverse_power_law(trampush_csv['RA_dis_m3_pmx_2'], *params)

# Calculating the corrected discharge values

guessworkRA = trampush_csv[['Wbf [m]', 'S [-]', 'RA_dis_m3_pmx_2']].astype(float)
guessworkRA.columns = ['width', 'slope', 'discharge']
trampush_csv['XGB_depth_RA'] = xgb_reg.predict(guessworkRA)

# Predicting the depth values for modeled discharge values

depth_predictions = pd.Series(xgb_reg.predict((Xtest[['width', 'slope', 'discharge']]))).astype(float)

# Predicting the depth values for the test data

guessworkM = trampush_csv[['Wbf [m]', 'S [-]', 'Qbf [m3/s]']].astype(float)
guessworkM.columns = ['width', 'slope', 'discharge']
trampush_csv['XGB_depth_M'] = xgb_reg.predict(guessworkM)

# Predicting the depth values for the manually recorded discharge values

guessworkRA_corr = trampush_csv[['Wbf [m]', 'S [-]', 'corrected_discharge']].astype(float)
guessworkRA_corr.columns = ['width', 'slope', 'discharge']
trampush_csv['XGB_depth_RA_corr'] = xgb_reg.predict(guessworkRA_corr)

#save out trampush csv and the df
trampush_csv.to_csv("data/figure2_data/TrampushDataCleanProcessed.csv", index=False)
df.to_csv("data/figure2_data/fig2_data.csv", index=False)

