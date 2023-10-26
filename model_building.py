import data_preparation
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from sklearn.metrics import r2_score, mean_absolute_error

nn_model = keras.Sequential()
nn_model.add(keras.layers.Input(shape=(data_preparation.x_train.shape[1]),))
nn_model.add(keras.layers.Dense(units=200, activation='relu', kernel_initializer="normal"))
nn_model.add(keras.layers.Dropout(rate=0.15))
nn_model.add(keras.layers.Dense(units=100, activation='relu', kernel_initializer="normal"))
nn_model.add(keras.layers.Dropout(rate=0.15))
nn_model.add(keras.layers.Dense(units=10, activation='relu', kernel_initializer="normal"))
nn_model.add(keras.layers.Dense(units=1))

callback = keras.callbacks.EarlyStopping(monitor='val_mae', patience=10, verbose=1, min_delta=4, mode="min")

opt = keras.optimizers.Adam(learning_rate=0.0001)

nn_model.compile(loss=keras.losses.MeanSquaredError(), optimizer=opt, metrics=["mae"])

history = nn_model.fit(data_preparation.x_train, data_preparation.y_train, epochs=100, validation_split=0.25, callbacks= [callback])

#nn_model.evaluate(x_test, y_test)

linear_reg = LinearRegression()
linear_reg_pca = LinearRegression()
lasso_reg = Lasso(alpha=0.5)
lasso_reg_pca = Lasso(alpha=0.5)
rrf = RandomForestRegressor(n_estimators=150, n_jobs=-1, verbose=2)
rrf_pca = RandomForestRegressor(n_estimators=150, n_jobs=-1, verbose=2)

#lasso_grid = {"alpha": [0.5, 1, 1.5]}
#rrf_grid = {"n_estimators" : [100],
#            "min_samples_split": [4,8],
#            'max_depth': [50, 100]}

#lasso_grid_search = GridSearchCV(lasso_reg, lasso_grid, scoring="r2", n_jobs=-1, verbose=3, cv=4)
#rrf_grid_search = GridSearchCV(rrf, rrf_grid, scoring="r2", n_jobs=-1, verbose=3, cv=2)

#grids = [lasso_grid_search, rrf_grid_search]

#Linear Regression
linear_reg.fit(data_preparation.x_train, data_preparation.y_train)
linear_reg.score(data_preparation.x_test, data_preparation.y_test)

linear_reg_pca.fit(data_preparation.x_train_pca, data_preparation.y_train)
linear_reg_pca.score(data_preparation.x_test_pca, data_preparation.y_test)

#Lasso Regression
lasso_reg.fit(data_preparation.x_train, data_preparation.y_train)
lasso_reg.score(data_preparation.x_test, data_preparation.y_test)

lasso_reg_pca.fit(data_preparation.x_train_pca, data_preparation.y_train)
lasso_reg_pca.score(data_preparation.x_test_pca, data_preparation.y_test)

##Random Forrest Regressor
rrf.fit(data_preparation.x_train, data_preparation.y_train)
rrf.score(data_preparation.x_test, data_preparation.y_test)

rrf_pca.fit(data_preparation.x_train_pca, data_preparation.y_train)
rrf_pca.score(data_preparation.x_test_pca, data_preparation.y_test)

from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor

gbr = GradientBoostingRegressor(verbose=2)
gbr_pca = GradientBoostingRegressor(verbose=2)
abr = AdaBoostRegressor(n_estimators=200)
abr_pca = AdaBoostRegressor(n_estimators=200)

#Gradient Bossting Regression
gbr.fit(data_preparation.x_train, data_preparation.y_train)
gbr.score(data_preparation.x_test, data_preparation.y_test)

gbr_pca.fit(data_preparation.x_train_pca, data_preparation.y_train)
gbr_pca.score(data_preparation.x_test_pca, data_preparation.y_test)

#AdaBoost Regression
abr.fit(data_preparation.x_train, data_preparation.y_train)
abr.score(data_preparation.x_test, data_preparation.y_test)

abr_pca.fit(data_preparation.x_train_pca, data_preparation.y_train)
abr_pca.score(data_preparation.x_test_pca, data_preparation.y_test)

print("Model Scores mit normalen Datensatz:")
print(f"Linear Regression {linear_reg.score(data_preparation.x_test, data_preparation.y_test)}")
print(f"Lasso Regression: {lasso_reg.score(data_preparation.x_test, data_preparation.y_test)}")
print(f"Random Forest Regressor: {rrf.score(data_preparation.x_test, data_preparation.y_test)}")
print(f"Gradient Boosting Regressor {gbr.score(data_preparation.x_test, data_preparation.y_test)}")
print(f"AdaBoost Regressor: {abr.score(data_preparation.x_test, data_preparation.y_test)}")

print("Model Scores mit PCA reduzierten Datensatz:")
print(f"Linear Regression {linear_reg_pca.score(data_preparation.x_test_pca, data_preparation.y_test)}")
print(f"Lasso Regression: {lasso_reg_pca.score(data_preparation.x_test_pca, data_preparation.y_test)}")
print(f"Random Forest Regressor: {rrf_pca.score(data_preparation.x_test_pca, data_preparation.y_test)}")
print(f"Gradient Boosting Regressor {gbr_pca.score(data_preparation.x_test_pca, data_preparation.y_test)}")
print(f"AdaBoost Regressor: {abr_pca.score(data_preparation.x_test_pca, data_preparation.y_test)}")

import pickle

pickle_out_lr= pickle.dump(linear_reg, open("lr_model.pkl", "wb"))
pickle_out_lasso = pickle.dump(lasso_reg, open("lasso_model.pkl", "wb"))
pickle_out__rrf = pickle.dump(rrf, open("rrf_model.pkl", "wb"))
pickle_out_gbr = pickle.dump(gbr, open("gbr_model.pkl", "wb"))
pickle_out_abr = pickle.dump(abr, open("abr_model.pkl", "wb"))
pickle_out_nn = pickle.dump(nn_model, open("nn_model.pkl", "wb"))

print("ready model_bulding")