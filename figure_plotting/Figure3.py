#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker

# Set color palette
palette = sns.color_palette("colorblind")

# Load data
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

# Set plot parameters
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 5
plt.rcParams['font.family'] = 'Helvetica'

# Create figure and gridspec
fig = plt.figure(figsize=(4, 2))  # Adjusted width to accommodate the CDF
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)

# Create subplots
ax1 = fig.add_subplot(gs[0])  # For scatter and lines
ax0 = fig.add_subplot(gs[1])  # For CDF, share the y-axis with ax1

# Plot CDF
sns.ecdfplot(data=df, x=df['gamma']*df['beta'], ax=ax0, color='k', linewidth=1.25)
ax0.set_xlabel(r'$\Lambda$', fontsize=7, labelpad=0) 
ax0.set_ylabel('CDF', fontsize=7, labelpad=-1)

# Calculate statistics
arithmetic_mean = np.mean(df['gamma']*df['beta'])
median_val = np.median(df['gamma']*df['beta'])

# Add vertical lines to the CDF
ax0.axvline(arithmetic_mean, color='k', linestyle='--', lw=0.6, label='Mean')
ax0.axvline(median_val, color='k', linestyle=':', lw=0.6, label='Median')

# Enhance legend
ax0.legend(fontsize=5)

# Customize ticks and grid
ax0.tick_params(axis='both', labelsize=6, direction="out", color='k', which='major',)
ax0.grid(False)

# Plot error bars and scatter plot
ax1.errorbar(df['gamma'], df['beta'], yerr=boot_df['beta_uncertainty'], xerr=[df['yerr_lower_relative'], df['yerr_upper_relative']], capsize=2.25, capthick=.4, ls='none', lw=.4, alpha=0.5, color='black', markeredgecolor='black', markersize=8, zorder=1)
sns.scatterplot(data=df, x=df['gamma'], y='beta', hue='geomorphology', hue_order=['Delta', 'Fan', 'Alluvial Plain'], s=40, edgecolor='k', ax=ax1, alpha=0.75, palette=palette)

# Define beta values and plot gamma_lambda for each theta value
beta = np.linspace(0.01, 12.5, 100)
thetas = [1, 2, 4]
lss = ['--', '-.', ':']
for i, theta in enumerate(thetas):
    gamma_lambda = (theta) / beta
    ax1.plot(gamma_lambda, beta, lw=.5, ls=lss[i], color='k', label=rf'$\Lambda = {theta}$')

# Customize ticks, grid, and labels
ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.set_ylabel(r'$\beta$', fontsize=7, rotation=0, labelpad=4)
ax1.set_xlabel(r'$\gamma$', fontsize=7, labelpad=-.8)
ax1.tick_params(axis='both', which='major', labelsize=6, direction="out", color='k')
ax1.grid(False)

# Set tick marks and formatters
ax0.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax0.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
ax0.xaxis.set_major_locator(ticker.FixedLocator([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
ax0.yaxis.set_major_locator(ticker.FixedLocator([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
ax1.xaxis.set_major_locator(ticker.FixedLocator([0.1, 1, 10, 100]))
ax1.yaxis.set_major_locator(ticker.FixedLocator([0.01, 0.1, 1, 10]))

# Disable minor ticks
ax0.xaxis.set_minor_locator(ticker.NullLocator())
ax0.yaxis.set_minor_locator(ticker.NullLocator())
ax1.xaxis.set_minor_locator(ticker.NullLocator())
ax1.yaxis.set_minor_locator(ticker.NullLocator())

# Customize spines
for spine in ax1.spines.values():
    spine.set_linewidth(.5)

# Add text labels
ax0.text(.05, 1.05, 'b', transform=ax0.transAxes, fontsize=6, fontweight='bold', va='top', ha='right')
ax1.text(.025, 1.05, 'a', transform=ax1.transAxes, fontsize=6, fontweight='bold', va='top', ha='right')

# Save and show plot
plt.margins(0, 0)
plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin to prevent cut-off labels

plt.savefig('figure_plotting/figures/figure3.png', dpi=300)
plt.savefig('figure_plotting/figures/figure3.pdf')
plt.show()
# %%