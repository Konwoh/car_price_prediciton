from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd 

df_prediction = pd.read_csv("data/df_preprocessed.csv")

cat_cols = df_prediction.select_dtypes("object").columns # Auswahl aller kategorischen Spalten 
num_cols = df_prediction.select_dtypes(["float64", "int64"]).columns # Auwahl aller numerischen Spalten

cols_without_price = list(num_cols[1:])
cols_without_price.remove("price")

prediciton_preprocessing = ColumnTransformer([
    ("scaler", StandardScaler(), cols_without_price),
    ("encoder", OneHotEncoder(), cat_cols)
],
remainder="passthrough")

x = df_prediction.drop(columns=["price"], axis=1)
y = df_prediction["price"]

x_pre = prediciton_preprocessing.fit_transform(x)

all_column_names = list(prediciton_preprocessing.named_transformers_['scaler'].get_feature_names_out(input_features=cols_without_price)) \
+ list(prediciton_preprocessing.named_transformers_['encoder'].get_feature_names_out(input_features=cat_cols)) + ["reg_year"]

df_prediction2 = pd.DataFrame(x_pre.toarray(), columns=all_column_names)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_prediction2, y, test_size=0.2, random_state=42)

x_train_pca = x_train.copy()
x_test_pca = x_test.copy()

from sklearn.decomposition import PCA

pca = PCA(n_components=20)

pca.fit(x_train_pca)

x_train_pca = pca.transform(x_train_pca)
x_test_pca = pca.transform(x_test_pca)

print("data fitting ready")