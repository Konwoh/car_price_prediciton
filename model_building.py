import data_fitting
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from sklearn.metrics import r2_score, mean_absolute_error
from nn_model import nn_model

nn_model = nn_model()
nn_model.compile_model()
callback = keras.callbacks.EarlyStopping(monitor='val_mae', patience=10, verbose=1, min_delta=4, mode="min")

history = nn_model.fit(data_fitting.x_train, data_fitting.y_train, epochs=100, validation_split=0.25, callbacks= [callback])

#nn_model.evaluate(x_test, y_test)

linear_reg = LinearRegression()
linear_reg_pca = LinearRegression()
lasso_reg = Lasso(alpha=0.5)
lasso_reg_pca = Lasso(alpha=0.5)
rrf = RandomForestRegressor(n_estimators=200, n_jobs=-1, verbose=2, max_depth=10)
rrf_pca = RandomForestRegressor(n_estimators=200, n_jobs=-1, verbose=2, max_depth=15)

lasso_grid = {"alpha": [0.5, 1, 1.5]}
#rrf_grid = {"n_estimators" : [100],
#            "min_samples_split": [4,8],
#            'max_depth': [50, 100]}

#lasso_grid_search = GridSearchCV(lasso_reg, lasso_grid, scoring="r2", n_jobs=-1, verbose=3, cv=4)
#rrf_grid_search = GridSearchCV(rrf, rrf_grid, scoring="r2", n_jobs=-1, verbose=3, cv=2)

#grids = [lasso_grid_search, rrf_grid_search]

#Linear Regression
linear_reg.fit(data_fitting.x_train, data_fitting.y_train)
linear_reg.score(data_fitting.x_test, data_fitting.y_test)

linear_reg_pca.fit(data_fitting.x_train_pca, data_fitting.y_train)
linear_reg_pca.score(data_fitting.x_test_pca, data_fitting.y_test)

#Lasso Regression
lasso_reg.fit(data_fitting.x_train, data_fitting.y_train)
lasso_reg.score(data_fitting.x_test, data_fitting.y_test)

lasso_reg_pca.fit(data_fitting.x_train_pca, data_fitting.y_train)
lasso_reg_pca.score(data_fitting.x_test_pca, data_fitting.y_test)

##Random Forrest Regressor
rrf.fit(data_fitting.x_train, data_fitting.y_train)
rrf.score(data_fitting.x_test, data_fitting.y_test)

rrf_pca.fit(data_fitting.x_train_pca, data_fitting.y_train)
rrf_pca.score(data_fitting.x_test_pca, data_fitting.y_test)

from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor

gbr = GradientBoostingRegressor(verbose=2)
gbr_pca = GradientBoostingRegressor(verbose=2)
abr = AdaBoostRegressor(n_estimators=200)
abr_pca = AdaBoostRegressor(n_estimators=200)

#Gradient Bossting Regression
gbr.fit(data_fitting.x_train, data_fitting.y_train)
gbr.score(data_fitting.x_test, data_fitting.y_test)

gbr_pca.fit(data_fitting.x_train_pca, data_fitting.y_train)
gbr_pca.score(data_fitting.x_test_pca, data_fitting.y_test)

#AdaBoost Regression
abr.fit(data_fitting.x_train, data_fitting.y_train)
abr.score(data_fitting.x_test, data_fitting.y_test)

abr_pca.fit(data_fitting.x_train_pca, data_fitting.y_train)
abr_pca.score(data_fitting.x_test_pca, data_fitting.y_test)

print("Model Scores mit normalen Datensatz:")
print(f"Linear Regression {linear_reg.score(data_fitting.x_test, data_fitting.y_test)}")
print(f"Lasso Regression: {lasso_reg.score(data_fitting.x_test, data_fitting.y_test)}")
print(f"Random Forest Regressor: {rrf.score(data_fitting.x_test, data_fitting.y_test)}")
print(f"Gradient Boosting Regressor {gbr.score(data_fitting.x_test, data_fitting.y_test)}")
print(f"AdaBoost Regressor: {abr.score(data_fitting.x_test, data_fitting.y_test)}")

print("Model Scores mit PCA reduzierten Datensatz:")
print(f"Linear Regression {linear_reg_pca.score(data_fitting.x_test_pca, data_fitting.y_test)}")
print(f"Lasso Regression: {lasso_reg_pca.score(data_fitting.x_test_pca, data_fitting.y_test)}")
print(f"Random Forest Regressor: {rrf_pca.score(data_fitting.x_test_pca, data_fitting.y_test)}")
print(f"Gradient Boosting Regressor {gbr_pca.score(data_fitting.x_test_pca, data_fitting.y_test)}")
print(f"AdaBoost Regressor: {abr_pca.score(data_fitting.x_test_pca, data_fitting.y_test)}")

import pickle

pickle_out_lr= pickle.dump(linear_reg, open("models/lr_model.pkl", "wb"))
pickle_out_lasso = pickle.dump(lasso_reg, open("models/lasso_model.pkl", "wb"))
pickle_out__rrf_pca = pickle.dump(rrf_pca, open("models/rrf_pca_model.pkl", "wb"))
pickle_out__rrf_pca = pickle.dump(rrf, open("models/rrf_pca_model.pkl", "wb"))
pickle_out_gbr = pickle.dump(gbr, open("models/gbr_model.pkl", "wb"))
pickle_out_abr = pickle.dump(abr, open("models/abr_model.pkl", "wb"))
nn_model.save_weights("models/model_weights.h5")

print("ready model_bulding")