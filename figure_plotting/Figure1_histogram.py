import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib

# Importing necessary libraries

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
plt.savefig('figure_plotting/figures/figure1_histogram.png', dpi=300)