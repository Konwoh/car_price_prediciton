import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Data/Ad_table (extra).csv")

df.columns = df.columns.str.lower()
df.columns = df.columns.str.strip()

df.drop(columns=["adv_id", "genmodel_id", "annual_tax"], inplace=True)

#Missing Values mit 0.0L auffüllen und L-Zeichen hinzufügen wo es fehlt, sodass danach mit apply nur die zahlen mit x[:-1] gesliced werden können
df["engin_size"].fillna("0.0L", inplace=True)
df["engin_size"] = df["engin_size"].apply(lambda x: x + "L" if "L" not in x else x)
df["engin_size"] = df["engin_size"].apply(lambda x: x[:-1])

obj_to_num = ["runned_miles", "engin_size", "top_speed"]

# Umwandeln der kategorischen Spalten in numerische Spalten 
for col in obj_to_num:
    df[col] = df[col].apply(lambda x: str(x).replace("nan", "0"))
    df[col] = df[col].apply(lambda x: str(x).split(" ")[0])
    df[col] = pd.to_numeric(df[col])
    
# Entfernen der Spalten die mit der Zielvariable Preis eine Korrelation von weniger als 0.1 haben
df.drop(list(df.corr()[np.abs(df.corr()["price"])< 0.1]["price"].index), axis=1, inplace=True)

# Funktion die zuerst bei allen kategorischen Spalten das mpg entfernt
# dann werden die Hersteller ermittelt, die kein Hersteller-Durchschnitt des jeweiligen Attributs zur Verfügung stellen können und diese werden entfernt
# danach wird über alle Auto-Namen iteriert und die NAN Values des besagten Attributs werden durch den Hersteller-Durchschnitt des Attributs aufgefüllt
# die restlichen NANs werden durch den Modell-Durchschnitt des besagten Attributs aufgefüllt, dies passiert in der zweiten for-Schleife

def fill_na_from_class(df, attr):
    if (df[attr].dtype == "object"):
        df[attr] = pd.to_numeric(df[attr].str.replace(" mpg", ""))
    maker_to_drop = list(df.groupby(by="maker")[attr].mean()[df.groupby(by="maker")[attr].mean().isna()].index)
    df.drop(df[df["maker"].isin(maker_to_drop)].index, axis=0, inplace=True)
    df["maker"] = df["maker"].apply(lambda x: x.replace(" ", ""))
    df["name"] = df["maker"] + " " + df["genmodel"]
    auto_list = df["name"].unique()
    
    for auto in auto_list:
        if auto in list(df.groupby(by="name")[attr].mean()[df.groupby(by="name")[attr].mean().isna()].index):
            df[df["name"] == auto] = df[df["name"] == auto].fillna(df[df["maker"] == auto.split(" ")[0]][attr].mean())
            
    missing_value_auto = list(df[df[attr].isna()]["name"].unique())
    
    for auto in auto_list:
        if auto in missing_value_auto:
            df[df["name"] == auto] = df[df["name"] == auto].fillna(df[df["name"] == auto][attr].mean())
            
    return df

df_mean = df.copy()

fill_na_from_class(df_mean, "engine_power")
fill_na_from_class(df_mean, "average_mpg")

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

cat_cols = df_mean.select_dtypes("object").columns # Auswahl aller kategorischen Spalten 
num_cols = df_mean.select_dtypes(["float64", "int64"]).columns # Auwahl aller numerischen Spalten

num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")

encoder = OneHotEncoder()

preprocessor = ColumnTransformer([
    ("num_imputer", num_imputer, num_cols),
    ("cat_imputer", cat_imputer, cat_cols)
])

df_new = pd.DataFrame(preprocessor.fit_transform(df_mean), columns = list(num_cols) + list(cat_cols))

# Entfernen von Ausreißern und 
df_new.drop(df_new[df_new["runned_miles"] > 1000000].index, axis=0, inplace=True) 
df_new.drop(df_new[df_new["runned_miles"] == 0.0].index, axis=0, inplace=True) 

df_new[num_cols] = df_new[num_cols].astype("float64")

for i in cat_cols:
    value_count = df_new[i].value_counts()
    drop_values = value_count[value_count < 20].index.tolist()
    df_new[i] = df_new[i].apply(lambda x: "Sonstige" if x in drop_values else x)
    

df_new.drop(columns=["name"], axis=1).to_csv("data/df_preprocessed.csv", index=False)

print("ready data_preparation")