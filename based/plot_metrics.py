import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import metrics

# Load the data (if needed)
X_test = pd.read_csv("data/train_test/based_us_Xtest_data.csv")
y_test = pd.read_csv("data/train_test/based_us_ytest_data.csv")


# Load the trained model
final_model = xgb.Booster()
final_model.load_model("based_us_no_fulton_with_early_stopping_combat_overfitting.ubj")


# Predict on the test set
depth_predictions = final_model.predict(xgb.DMatrix(X_test))

# Evaluation Metrics
mae = metrics.mean_absolute_error(y_test, depth_predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test, depth_predictions))
r2 = metrics.r2_score(y_test, depth_predictions)

print("MAE: %f" % mae)
print("RMSE: %f" % rmse)
print("R2: %f" % r2)

# Set up plot style
plt.figure(figsize=(10, 10))
sns.set_style('whitegrid')
plt.axis('equal')
plt.tight_layout()

# Scatter plot for predicted vs actual values
sns.scatterplot(x=y_test, y=depth_predictions, color='red', edgecolor='k', s=100)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', label=f'1:1 line, n = {len(y_test)}', lw=3.5)
plt.xlabel('Measured Channel Depth (m)')
plt.ylabel('Predicted Channel Depth (m)')
plt.yscale('log')
plt.xscale('log')

# Save the plots
plt.savefig("data/img/BASED_validation.png", dpi=300)
plt.savefig("data/img/BASED_validation.pdf")
plt.show()  # Display the plot
