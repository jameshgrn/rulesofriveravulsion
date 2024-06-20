# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import seaborn as sns
import xgboost as xgb
from scipy.optimize import curve_fit
import pickle
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib import rcParams
from matplotlib import rc
import matplotlib.font_manager
from matplotlib.font_manager import FontProperties
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
sns.set_context('paper', font_scale=1.0)
sns.set_style('whitegrid')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 5



trampush_csv = pd.read_csv("data/BASED_model/TrampushDataClean.csv").dropna(subset=["RA_dis_m3_pmx_2", "Qbf [m3/s]"])

ytest = pd.read_csv("data/BASED_model/train_test/based_us_ytest_data_sans_trampush.csv")
Xtest = pd.read_csv("data/BASED_model/train_test/based_us_Xtest_data_sans_trampush.csv")

# Loading test data and data to be processed
# removing data that is mismatched with RiverATLAS locations
trampush_csv = trampush_csv.query("Location != 'Ditch Run near Hancock'")
trampush_csv = trampush_csv.query("Location != 'Potomac River Tribuatary near Hancock. Md.'")
trampush_csv = trampush_csv.query("Location != 'Clark Fork tributary near Drummond'")
trampush_csv = trampush_csv.query("Location != 'Richards Creek near Libby MT'")
trampush_csv = trampush_csv.query("Location != 'Ohio Brush Creek near West Union'")
trampush_csv = trampush_csv[trampush_csv["Location"] != "Floyd's Fork at Fisherville"]

xgb_reg = xgb.XGBRegressor()
xgb_reg.load_model("data/BASED_model/models/based_us_sans_trampush_early_stopping_combat_overfitting.ubj")

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

with open('data/BASED_model/power_law_params.pickle', 'wb') as f:
    pickle.dump(params, f)

# Saving power law parameters to a file

with open('data/BASED_model/inverse_power_law_params.pickle', 'wb') as f:
    pickle.dump(params, f)

# Saving inverse power law parameters to a file

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

# Define marker size variable
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
marker_size = 12


fig = plt.figure(figsize=(7, 4), constrained_layout=False)
gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1])

# Create subplots in desired positions
axs = []
axs.append(fig.add_subplot(gs[0, 0]))  # Top left
axs.append(fig.add_subplot(gs[0, 2]))  # Top right
axs.append(fig.add_subplot(gs[1, 0]))  # Middle bottom left
axs.append(fig.add_subplot(gs[1, 1]))  # Middle bottom center
axs.append(fig.add_subplot(gs[1, 2]))  # Middle bottom right

# Load data
ytest = pd.read_csv("data/BASED_model/train_test/based_us_ytest_data_sans_trampush.csv")
Xtest = pd.read_csv("data/BASED_model/train_test/based_us_Xtest_data_sans_trampush.csv")

# # Ensure ytest is a Series
ytest = ytest.iloc[:, 1]

# Calculate pred_depth
pred_depth = 0.31 * Xtest['discharge'] ** 0.38
axs[0].grid(zorder=0)
axs[0].scatter(ytest, pred_depth, color='green', edgecolor='k', s=12, zorder=3)
# Calculate common limits
common_min = min(pred_depth.min(), ytest.min())
common_max = max(pred_depth.max(), ytest.max())
axs[0].plot([common_min, common_max], [common_min, common_max], 'k--', lw=1.5, label=f'1:1 line, n = {len(ytest)}')
axs[0].set_xlabel('Measured Depth compiled by Deal et. al', fontsize=7)
axs[0].set_ylabel('Power Law Predicted Depth', fontsize=7)
axs[0].grid(True)
axs[0].set_aspect('equal', adjustable='box')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].legend()
axs[0].text(0.01, 1.02, 'A', transform=axs[0].transAxes, size=7, weight='bold')

# Scatter plot for predicted vs actual values
axs[1].scatter(ytest, depth_predictions, color='red', edgecolor='k', s=marker_size)
axs[1].set_xlabel('Measured Channel Depth (m)', fontsize=7)
axs[1].set_ylabel('Predicted Channel Depth (m)', fontsize=7)
axs[1].set_yscale('log')
axs[1].set_xscale('log')
min_val, max_val = min(min(ytest.values), min(depth_predictions.values)), max(max(ytest.values), max(depth_predictions.values))
axs[1].set_xlim(min_val, max_val)
axs[1].set_ylim(min_val, max_val)
# After setting your limits, you can extract these values
xlims = axs[1].get_xlim()
ylims = axs[1].get_ylim()
# Get minimum and maximum values considering both x and y limits
common_min = max(min(xlims), min(ylims))  # Using 'max' here to consider the highest minimum value
common_max = min(max(xlims), max(ylims))  # Using 'min' here to consider the lowest maximum value
# Now plot the line with these values
axs[1].plot([common_min, common_max], [common_min, common_max], 'k--', label=f'1:1 line, n = {len(ytest)}', lw=1.5)
axs[1].set_aspect('equal', adjustable='box')
axs[1].legend()
axs[1].text(0.01, 1.02, 'B', transform=axs[1].transAxes, size=7, weight='bold')


# Scatter plot for the first subplot
axs[2].scatter(trampush_csv['Hbf [m]'], trampush_csv['XGB_depth_RA'], alpha=0.8, c='r', edgecolor='k', label='RiverATLAS $Q_{max}$', s=marker_size)
axs[2].scatter(trampush_csv['Hbf [m]'], trampush_csv['XGB_depth_M'], alpha=0.8, c='b', edgecolor='k', label='Measured $Q_{bf}$', s=marker_size)
axs[2].plot([0, 3], [0, 3], linestyle='--', color='gray', label='1:1 Line')
axs[2].set_xlabel('Measured $H_{bf}$ from Trampush et. al', fontsize=7)
axs[2].set_ylabel('BASED Depth Prediction [m]', fontsize=7)
axs[2].grid(True)
axs[2].set_xlim(0, 3)
axs[2].set_ylim(0, 3)
axs[2].legend()
axs[2].set_xlim(0, 3)
axs[2].set_ylim(0, 3)
axs[2].set_aspect('equal', adjustable='box')
axs[2].text(0.01, 1.02, 'C', transform=axs[2].transAxes, size=7, weight='bold')

# Scatter plot for the second subplot
axs[3].scatter(trampush_csv['Qbf [m3/s]'], trampush_csv['RA_dis_m3_pmx_2'], s=marker_size, color='w', edgecolor='k', linewidth=0.5)
min_val = min(trampush_csv['Qbf [m3/s]'].min(), trampush_csv['RA_dis_m3_pmx_2'].min())
max_val = max(trampush_csv['Qbf [m3/s]'].max(), trampush_csv['RA_dis_m3_pmx_2'].max())
axs[3].set_xlim(min_val, max_val)
axs[3].set_ylim(min_val, max_val)
axs[3].plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', label='1:1')
axs[3].set_xlabel('Qbf [m3/s]', fontsize=7)
axs[3].set_ylabel('RiverATLAS $Q_{max}$ [m3/s]', fontsize=7)
axs[3].grid(True)
axs[3].set_aspect('equal', adjustable='box')
axs[3].set_xscale('log')
axs[3].set_yscale('log')
axs[3].text(0.01, 1.02, 'D', transform=axs[3].transAxes, size=7, weight='bold')
axs[3].plot(x_fit, y_fit, color='black', ls='-.', label='Power law Fit')
axs[3].legend()

# Create a new subplot for the corrected values scatter plot
axs[4].scatter(trampush_csv['Hbf [m]'], trampush_csv['XGB_depth_M'], alpha=0.8, c='b', edgecolor='k', label='Measured $Q_{bf}$', s=marker_size)
axs[4].scatter(trampush_csv['Hbf [m]'], trampush_csv['XGB_depth_RA_corr'], alpha=0.8, c='r', edgecolor='k', label='Corrected RA $Q_{max}$', s=marker_size)
axs[4].plot([0, 3], [0, 3], linestyle='--', color='gray')
axs[4].set_xlabel('Measured $H_{bf}$ from Trampush et. al', fontsize=7)
axs[4].set_ylabel('BASED Depth Prediction [m]', fontsize=7)
axs[4].grid(True)
axs[4].legend()
axs[4].set_xlim(0, 3)
axs[4].set_ylim(0, 3)
axs[4].set_aspect('equal', adjustable='box')
axs[4].text(0.01, 1.02, 'E', transform=axs[4].transAxes, size=7, weight='bold')

plt.subplots_adjust(top=0.95, wspace=0.25, hspace=0.25)

# Adjust layout
for ax in axs:
    ax.legend(fontsize=5)
    ax.tick_params(axis='both', labelsize=6, pad=-1.5)

    # Adjust label padding for x and y labels by -2
    ax.set_xlabel(ax.get_xlabel(), labelpad=-1)
    ax.set_ylabel(ax.get_ylabel(), labelpad=-1)

# Save the figure
plt.savefig("figure_plotting/figures/extended_data_figs/extended_data_figure_3.png", dpi=300, bbox_inches='tight')
plt.savefig("figure_plotting/figures/extended_data_figs/extended_data_figure_3.pdf", dpi=300, bbox_inches='tight')


fig, ax = plt.subplots(figsize=(3, 3))
scatter = ax.scatter(trampush_csv['Hbf [m]'], trampush_csv['XGB_depth_M'], alpha=0.8, c='b', edgecolor='k', label='Validation Plot', s=marker_size)
ax.set_xlabel('Measured Depth (m)')
ax.set_ylabel('XGBoost Pred. Depth (m)')
ax.set_title('Validation Plot')
min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 Line')
ax.legend()
plt.tight_layout()
plt.savefig("/Users/jakegearon/projects/misc/valplot.pdf", bbox_inches='tight')
plt.show()

# %%
