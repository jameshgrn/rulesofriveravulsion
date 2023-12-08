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
plt.savefig('figures/FIGURE2.png', dpi=300)
plt.show()