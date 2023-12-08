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

with open('data/power_law_params.pickle', 'wb') as f:
    pickle.dump(params, f)

# Saving power law parameters to a file

with open('data/inverse_power_law_params.pickle', 'wb') as f:
    pickle.dump(params, f)

# Saving inverse power law parameters to a file

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
