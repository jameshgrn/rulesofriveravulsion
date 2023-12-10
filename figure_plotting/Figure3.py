import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker



palette = sns.color_palette("colorblind")



df = pd.read_csv('data/figure2_data/fig2_data.csv')
trampush_csv = pd.read_csv("data/figure2_data/TrampushDataCleanProcessed.csv")
boot_df = pd.read_csv('data/figure2_data/fig2_data_boot.csv')

# Custom formatter function
def custom_formatter(x, pos):
    # Check if the value is less than 1 and format it with an additional decimal place
    if x < 1 and x >= 0.1:
        return '{:.1f}'.format(x)  # 1 decimal places for values less than 1
    elif x < 0.1 and x >= 0.01:
        return '{:.2f}'.format(x) # 2 decimal places for values less than .1
    elif x == 0:
        return '{:.0f}'.format(x)
    else:
        return '{:.0f}'.format(x)  # 0 decimal places for all other values

sns.set_context('paper', font_scale=1.0)
sns.set_style('whitegrid')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 5

fig = plt.figure(figsize=(5, 3))  # Adjusted width to accommodate the CDF
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.2)

ax1 = fig.add_subplot(gs[0])  # For scatter and lines
ax0 = fig.add_subplot(gs[1])  # For CDF, share the y-axis with ax1

# Plot histogram on the top axis
sns.ecdfplot(data=df, x=df['gamma']*df['beta'], ax=ax0, color='k', linewidth=1.25)
ax0.set_xlabel(r'$\Lambda$', fontsize=7, labelpad=0) 
ax0.set_ylabel('CDF')
ax0.tick_params(axis='y', labelsize=5)  # No y-axis labels on the CDF
ax0.tick_params(axis='x', labelsize=5)
# Calculate the arithmetic mean, geometric mean, and median
arithmetic_mean = np.mean(df['gamma']*df['beta'])
median_val = np.median(df['gamma']*df['beta'])

# Add vertical lines to the histogram
ax0.axvline(arithmetic_mean, color='k', linestyle='--', lw=0.6, label='Mean')
#ax0.axvline(geometric_mean_val, color='k', linestyle='-.', lw=0.6, label='Geometric Mean')
ax0.axvline(median_val, color='k', linestyle=':', lw=0.6, label='Median')

# Enhance legend to include new lines
ax0.legend(fontsize=5)

#ax0.set_xticks([])  # Remove xticks for the histogram
ax0.set_ylabel('CDF', fontsize=7)
ax0.tick_params(axis='both', labelsize=5, pad=-2)
ax1.errorbar(df['gamma'], df['beta'], yerr=boot_df['beta_uncertainty'], xerr=[df['yerr_lower_relative'], df['yerr_upper_relative']], capsize=2.25, capthick=.4, ls='none', lw=.4, alpha=0.5, color='black', markeredgecolor='black', markersize=8, zorder=1)
# Scatter plot
sns.scatterplot(data=df, x=df['gamma'], y='beta', hue='geomorphology', hue_order=['Delta', 'Fan', 'Alluvial Plain'], s=40, edgecolor='k', ax=ax1, alpha=0.75, palette=palette)
# ax1.set_ylim(0, 5)
# ax1.set_xlim(0, 35)
# Define beta values
beta = np.linspace(0.01, 5.5, 100)
# Calculate and plot gamma_lambda for each theta value
thetas = [1, 2, 4]
colors = ['#003366', '#8B0000', '#4DB6AC']
lss = ['--', '-.', ':']
for i, theta in enumerate(thetas):
    gamma_lambda = (theta) / beta
    ax1.plot(gamma_lambda, beta, lw=.5, ls=lss[i], color='k', label=rf'$\Lambda = {theta}$')

ax1.grid()
ax1.legend(fontsize=5)
ax1.set_ylabel(r'$\beta$', fontsize=7, rotation=0, labelpad=10)
ax1.set_xlabel(r'$\gamma$', fontsize=7)
#ax1.set_xlim(-1, 35)

ax1.grid()
ax1.tick_params(axis='both', which='major', labelsize=7)
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))


for spine in ax1.spines.values():
    spine.set_linewidth(.5)
ax1.tick_params(axis='both', labelsize=5, pad=-2)

ax0.text(.035, 1.035, 'B', transform=ax0.transAxes, 
         fontsize=6, fontweight='bold', va='top', ha='right')

# Label for the scatter plot
ax1.text(.025, 1.035, 'A', transform=ax1.transAxes, 
         fontsize=6, fontweight='bold', va='top', ha='right')

ax1.set_yticks([.01, 0.1, 1, 10])
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
plt.tight_layout()
plt.margins(0, 0)
plt.savefig('figure_plotting/figures/figure3.png', dpi=300)
