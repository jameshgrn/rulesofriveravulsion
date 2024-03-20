# %%
import matplotlib.ticker as ticker
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
import os

meanprops = {
    "marker": "D",
    "markerfacecolor": "white",
    "markeredgecolor": "black",
    "markersize": "3",
    "markeredgewidth": 0.25
}

flierprops = {
    "marker": "^",
    "markerfacecolor": "white",
    "markeredgecolor": "black",
    "markersize": "3",
    "markeredgewidth": 0.5
}
fontsize = 7
palette = sns.color_palette("colorblind")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(font_scale=1.0, style="white")

df = pd.read_csv('data/figure2_data/fig2_data.csv')
trampush_csv = pd.read_csv("data/figure2_data/TrampushDataCleanProcessed.csv")
df_hist = pd.read_excel('data/manuscript_data/Supplementary Table 1.xlsx')
df_hist['normalized_distance'] = df_hist['normalized_distance'].astype(float)

n_simulations = 10000

ridge_elev_error = 0.6
floodplain_elev_error = 0.6
water_surface_elev_error = 0.25
xgb_depth_error = 1.0

superelevation_column = 'beta'
uncertainty_column = 'beta_uncertainty'

for index, row in df.iterrows():
    all_superelevations = []

    for _ in range(n_simulations):
        i = np.random.choice([1, 2, 3])

        ridge_height = row[f'are{i}_m'] - row[f'fpe{i}_m']
        sampled_ridge_height = ridge_height + np.random.triangular(-ridge_elev_error, 0, ridge_elev_error)
        sampled_water_surface_elevation = row[f'wse{i}_m'] + np.random.triangular(-water_surface_elev_error, 0, water_surface_elev_error)
        sampled_xgb_depth = row['xgb_depth'] + np.random.triangular(-xgb_depth_error, 0, xgb_depth_error)

        if row['method_used'] == 1:
            superelevation = sampled_ridge_height / sampled_xgb_depth
        elif row['method_used'] == 2:
            superelevation = sampled_ridge_height / (row[f'are{i}_m'] - sampled_water_surface_elevation)
        elif row['method_used'] == 3:
            superelevation = sampled_ridge_height / (row[f'are{i}_m'] - (sampled_water_surface_elevation - sampled_xgb_depth))

        all_superelevations.append(superelevation)

    df.at[index, superelevation_column] = np.mean(all_superelevations)
    df.at[index, uncertainty_column] = np.std(all_superelevations)

boot_df = df  # Now df contains all the needed columns.
boot_df.to_csv("data/figure2_data/fig2_data_boot.csv", index=False)

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


def plot_scatter_with_lowess(ax, data, x_col, y_col, frac=1, len_boot=5000, color_data=None, palette=palette, edgecolor='black'):
    # Extract data from DataFrame
    x = data[x_col]
    y = np.log(data[y_col])  # Log-transform y values

    # If color_data is specified, use hue in scatterplot
    if color_data:
        sns.scatterplot(x=data[x_col], y=data[y_col], hue=data[color_data], s=20, ax=ax, palette=palette, hue_order=['Delta', 'Fan', 'Alluvial Plain'], edgecolor=edgecolor)
    else:
        sns.scatterplot(x=data[x_col], y=data[y_col], ax=ax)

    # Fit the LOWESS model to the log-transformed data
    lowess_results = lowess(y, x, frac=frac, it=5)
    y_predicted = lowess_results[:, 1]
    residuals = y - y_predicted

    # Perform bootstrapping to estimate the confidence intervals in log space
    y_bootstrapped = np.empty((len_boot, len(y)))
    for i in range(len_boot):
        residuals_bootstrap = resample(residuals)
        y_bootstrapped[i] = y_predicted + residuals_bootstrap

    # Calculate the lower and upper bounds of the confidence intervals in log space
    y_ci_lower = np.percentile(y_bootstrapped, 2.5, axis=0)
    y_ci_upper = np.percentile(y_bootstrapped, 97.5, axis=0)

    # Transform the predicted values and confidence intervals back to linear space for plotting
    y_predicted = np.exp(y_predicted)
    y_ci_lower = np.exp(y_ci_lower)
    y_ci_upper = np.exp(y_ci_upper)

    # Plot the LOWESS fit and the confidence intervals
    ax.plot(lowess_results[:, 0], y_predicted, color='red', linestyle='--', linewidth=1.)
    ax.fill_between(lowess_results[:, 0], y_ci_lower, y_ci_upper, color='lightgray', alpha=0.3)  # Lighter gray

    # Formatting and labeling
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend().set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

    return ax

# Then include the custom function before plotting:
def format_func(value, tick_number):
    return "{:.0f}".format(value)
    
def plot_boxplot(ax, x_data, y_data, order, ylabel=''):
    sns.boxplot(data=df, x=x_data, y=y_data, hue='geomorphology', showfliers=False, flierprops=flierprops,
                showmeans=True, meanprops=meanprops,
                order=order, hue_order=['Delta', 'Fan', 'Alluvial Plain'], ax=ax, palette=palette, linewidth=0.5, width=0.75)

    ax.grid(axis='y', visible=True)
    ax.set_ylabel(ylabel, labelpad=10, rotation=0)
    labels = ax.get_xticklabels()
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(labels, fontsize=5, ha='center')
    

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
sns.histplot(data=df_hist, x='normalized_distance', kde=False, fill=True,
             alpha=1, edgecolor='black', stat="percent", binwidth=.05, color='#56b4e9', linewidth=.65, ax=ax_new_hist) #'#4FA0CA'
ax_new_hist.grid(False)
ax_new_hist.set_ylim(0, 45)
ax_new_hist.set_xlabel('$X_N$', fontsize=6, labelpad=-3)
ax_new_hist.set_ylabel('% Occurrence', fontsize=6, labelpad=-.5)
ax_new_hist.tick_params(axis='both', which='major', labelsize=6, width=0, length=2)
sns.despine(ax=ax_new_hist)

ax0 = fig.add_subplot(gs1[1:4, 2])
ax0.errorbar(boot_df['spr'], boot_df['beta'], yerr=boot_df['beta_uncertainty'], capsize=3.5, capthick=0.5, ls='none', lw=.5, color='black', markeredgecolor='black', markersize=8, zorder=1)
plot_scatter_with_lowess(ax0, df, 'spr', 'beta', color_data='geomorphology', palette=palette)
ax0.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
ax0.set_xlabel('')
ax0.set_ylabel('')
ax0.set_xscale('log')

ax1 = fig.add_subplot(gs1[4:7, 2])
ax1.errorbar(df['spr'], df['gamma'], yerr=[df['yerr_lower_relative'], df['yerr_upper_relative']], capsize=3.5, capthick=0.5, ls='none', lw=.5, color='black', markeredgecolor='black', markersize=8, zorder=1)
plot_scatter_with_lowess(ax1, df, 'spr', 'gamma', color_data='geomorphology', palette=palette)
ax1.set_xlabel(r'proximal and steep $\longleftarrow SPR^{\star} \longrightarrow$ distal and gentle', fontsize=fontsize, labelpad=1)
ax1.set_xscale('log')
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

ax3 = fig.add_subplot(gs1[1:4, 1])
plot_boxplot(ax3, 'geomorphology', 'beta', ['Fan', 'Alluvial Plain', 'Delta'])
ax3.set_xlabel('')
ax3.set_ylabel('')

ax4 = fig.add_subplot(gs1[4:7, 1])
plot_boxplot(ax4, 'geomorphology', 'gamma', ['Fan', 'Alluvial Plain', 'Delta'])
ax4.set_xlabel('')
ax4.set_ylabel('')
ax4.set_xlabel('Geomorphology', fontsize=fontsize)

ax6 = fig.add_subplot(gs1[1:4, 0])
sns.histplot(data=df, y='beta', bins=15, edgecolor='k', color='pink', alpha=1, stat="probability", ax=ax6, zorder=3, linewidth=.5)
ax6.set_ylabel(r'$\beta$', rotation=0, fontsize=fontsize)
ax6.set_yticks([0, 1, 2, 3, 4, 5])
ax6.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax6.grid(False)
ax6.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
ax6.set_xlabel('')
ax6.axhspan(.5, 1.1, color='#BDBDBD', alpha=.6, fill=True, zorder=1)
ax6.set_ylim(0, 5.5)
median_superelevation = df['beta'].median()
ax6.axhline(y=median_superelevation, color='black', linestyle='-.', linewidth=.75, zorder=4)

ax7 = fig.add_subplot(gs1[4:7, 0])
ax7.grid(False)
sns.histplot(data=df, y='gamma', bins=15, edgecolor='k', color='pink', alpha=1, stat="probability", ax=ax7, zorder=3, linewidth=.5)
ax7.set_ylabel(r'$\gamma$', rotation=0, fontsize=fontsize)
ax7.set_xlabel('Probability', fontsize=fontsize)
ax7.axhspan(3, 10, color='#BDBDBD', alpha=0.6, zorder=1)
ax7.set_ylim(-.5, 35)
median_XS_DS = df['gamma'].median()
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
plt.savefig('figure_plotting/figures/figure2.png', dpi=300)
plt.savefig('figure_plotting/figures/figure2.pdf')
plt.show()

# %%
