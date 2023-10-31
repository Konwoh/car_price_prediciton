import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_begin = pd.read_csv("data/Ad_table (extra).csv") #Ursprügnlciher datensatz
df = pd.read_csv("data/df_preprocessed.csv") #vorgefertigeter Datensatz (Resultat von data_fitting.py)

num_cols = df.select_dtypes(["float64", "int64"]).columns #Numerische Attribute des Datensatzes
cat_cols = df.select_dtypes(["object"]).columns #Kategorische Attribute des Datensatzes

## Erste Visualisierung
plt.figure(figsize=(12, 7))
(df_begin.isna().sum()/df_begin.shape[0]).sort_values(ascending=False).plot(kind="bar")
plt.title("Prozentualer Anteil an Null-Werten der verschiedenen Attribute des Datensatzes")
plt.xlabel("Attribute Datensatz")
plt.ylabel("Prozent")
#plt.show()
plt.savefig("Visualizations/missing_values.png")
plt.clf()

## Zweite Visualisierung
sns.heatmap(df_begin.select_dtypes(exclude="object").corr(), annot=True, annot_kws={"fontsize":6}, cbar=False)
plt.title("Korrelations-Heatmap")
#plt.show()
plt.savefig("Visualizations/korr_heatmap.png")
plt.clf()

## Dritte Visualisierung
fig3, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(15,10))
axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
fig3.suptitle('Verteilung aller numerischen Datenwerte')
for col, ax in zip(num_cols, axes):
    ax.hist(df[col], bins=100)
    ax.set_title(col)
#plt.show()
plt.savefig("Visualizations/num_verteilung.png")
plt.clf()

##Vierte Visualisierung
fig4, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15,12))
axes2 = [ax1, ax2, ax3, ax4]
for cat, ax in zip(cat_cols[2:], axes2):
    sns.countplot(data = df,x= cat, ax=ax)
    ax.tick_params(axis='x', labelrotation=60)
#plt.show()
plt.savefig("Visualizations/cat_verteilung.png")
plt.clf()

##Fünfte Visualisierung
plt.figure(figsize=(12, 9))
df.maker.value_counts().plot(kind="bar")
plt.title("Anzahl Autos pro Hersteller im Datensatz")
plt.xlabel("Hersteller", fontsize=6)
plt.ylabel("Anzahl Autos", fontsize=6)
#plt.show()
plt.savefig("Visualizations/anzahl_autos.png")
plt.clf()

##Sechste Visualisierung      
mean_maker = df.groupby(by=["maker"])["price"].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 9))
(mean_maker[mean_maker < 1000000]).plot(kind="bar")
plt.title("Durchschnittlicher Preis je Hersteller(Auswahl)")
plt.xlabel("Hersteller")
plt.ylabel("Preis")
#plt.show()
plt.savefig("Visualizations/mean_price.png")
plt.clf()

##Siebte Visualisierung
plt.figure(figsize=(12, 7))
df_maker = df[df["maker"].isin(list(df["maker"].value_counts().head().index))]
df_maker = df_maker[df_maker["bodytype"].isin(["Saloon", "Convertible", "SUV", "Estate", "Coupe", "Hatchback"])]
plot_test = df_maker.groupby(by=["reg_year", "maker"])["price"].mean()
plot_test_df = pd.DataFrame(plot_test).reset_index()
plot_test_df = plot_test_df[plot_test_df.reg_year > 1990]
sns.lineplot(x=plot_test_df["reg_year"], y=plot_test_df["price"], hue=plot_test_df["maker"])
plt.xlabel("Zulassungsjahr")
plt.ylabel("Preis")
plt.title("Durchschnittlicher Preis der größten Hersteller über Zeit")
#plt.show()
plt.savefig("Visualizations/price_over_time.png")
plt.clf()

## Achte Visualisierung
plt.figure(figsize=(12, 7))
scatter_df = df.groupby(by=["maker", "genmodel"])[["runned_miles", "price"]].mean().reset_index()
filtered = scatter_df[(scatter_df["price"] < 100000) & (scatter_df["runned_miles"] < 400000) & (scatter_df["maker"].isin(scatter_df["maker"].value_counts().head().index))]
sns.scatterplot(data=filtered, y="price", x="runned_miles", hue="maker")
plt.title("Verhältnis Laufleistung und Preis der größten Hersteller")
plt.xlabel("Laufleistung")
plt.ylabel("Preis")
#plt.show()
plt.savefig("Visualizations/miles_price_scatter.png")
