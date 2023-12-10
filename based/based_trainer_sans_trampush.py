import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV

# Load the data
df = pd.read_csv("data/BASED_model/based_input_data.csv").drop("Unnamed: 0", axis=1)
df = df[df['discharge'] > 0]
# Remove rows where 'source' contains 'Trampush' case-insensitively as they are our validation dataset
df = df[~df['source'].str.contains('Trampush', case=False, na=False)]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop("depth", axis=1), df["depth"], test_size=0.20, random_state=42, shuffle=True)
X_train = X_train.filter(["width", "slope", "discharge"], axis=1)
X_test = X_test.filter(["width", "slope", "discharge"], axis=1)

# Define the hyperparameter search space
param_space_combat_overfitting = {
    'max_depth': [3, 4, 5],
    'min_child_weight': [5, 10, 15],
    'reg_alpha': [1, 2, 5],
    'reg_lambda': [1, 2, 5],
    "subsample": [0.1, 0.25, 0.5, 0.75, 0.9],
    "colsample_bytree": [0.1, 0.2, 0.5, 0.7, 0.8],
    'learning_rate': [0.01, 0.05, 0.1],
}

# BayesSearchCV for finding the best hyperparameters
opt = BayesSearchCV(
    xgb.XGBRegressor(objective='reg:squarederror'),
    param_space_combat_overfitting,
    scoring='neg_mean_absolute_error',
    n_iter=75,
    n_jobs=6,
    cv=5,
    verbose=1,
    random_state=42
)

opt.fit(X_train, y_train)

# Get the best parameters
best_params = opt.best_params_

# Initialize DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)

# Define the early stopping rounds
early_stopping_rounds = 20

# Perform cross-validation with early stopping
cv_results = xgb.cv(
    best_params,
    dtrain,
    num_boost_round=1000,
    nfold=5,
    metrics=['mae'],
    early_stopping_rounds=early_stopping_rounds,
    seed=42,
    verbose_eval=True
)

# Get the best number of boosting rounds
n_best_trees = cv_results['test-mae-mean'].idxmin() + 1

# Train the final model with the best number of boosting rounds and best hyperparameters
final_model = xgb.train(best_params, dtrain, num_boost_round=n_best_trees)

# Save the model
final_model.save_model("data/BASED_model/models/based_us_sans_trampush_early_stopping_combat_overfitting.ubj")
X_test.to_csv("data/BASED_model/train_test/based_us_Xtest_data_sans_trampush.csv")
y_test.to_csv("data/BASED_model/train_test/based_us_ytest_data_sans_trampush.csv")
X_train.to_csv("data/BASED_model/train_test/based_us_Xtrain_data_sans_trampush.csv")
y_train.to_csv("data/BASED_model/train_test/based_us_ytrain_data_sans_trampush.csv")
