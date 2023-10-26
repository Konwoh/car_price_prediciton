import pickle
import joblib
import pandas as pd
import streamlit as st
#from data_preparation import prediciton_preprocessing
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import data_fitting
import seaborn as sns

#laden der trainerten Modelle
loaded_model_lr = pickle.load(open("models/lr_model.pkl", "rb"))
loaded_model_lasso = pickle.load(open("models/lasso_model.pkl", "rb"))
loaded_model_rrf = joblib.load(open("models/rrf.joblib", "rb"))
loaded_model_gbr = pickle.load(open("models/gbr_model.pkl", "rb"))
loaded_model_abr = pickle.load(open("models/abr_model.pkl", "rb"))
loaded_model_nn = pickle.load(open("models/nn_model.pkl", "rb"))

#laden der Datensätze
df_begin = pd.read_csv("data/Ad_table (extra).csv") #Ursprügnlciher datensatz
df = pd.read_csv("data/df_preprocessed.csv") #vorgefertigeter Datensatz (Resultat von data_preparation.py)

num_cols = df.select_dtypes(["float64", "int64"]).columns #Numerische Attribute des Datensatzes
cat_cols = df.select_dtypes(["object"]).columns #Kategorische Attribute des Datensatzes

def auto_price_predicition(input_data, model):
    df_test = pd.DataFrame(list(input_data.values())).T
    df_test.columns = df.drop(columns="price").columns
    input_pre = data_fitting.prediciton_preprocessing.transform(df_test)
    
    if model == "Linear Regression":
        return loaded_model_lr.predict(input_pre)
    elif model == "Lasso Regression":
        return loaded_model_lasso.predict(input_pre)
    elif model == "Random Forest Regressor":
        return loaded_model_rrf.predict(input_pre)
    elif model == "Gradient Boosting Regressor":
        return loaded_model_gbr.predict(input_pre)
    elif model == "AdaBoost Regressor":
        return loaded_model_abr.predict(input_pre)
    elif model == "Neural Network":
        return loaded_model_nn.predict(input_pre)

def prediction():
    df_pred = df.drop(columns="price")
    columns = list(df_pred.dtypes.index)
    dtype = df_pred.dtypes
    input_data = {}
    
    # Definieren Sie Ihre Spalten in Gruppen, um sie in drei Spalten anzuzeigen
    column_groups = [columns[:5], columns[5:10], columns[10:]]
    col1, col2, col3 = st.columns(3)
    
    for group in column_groups:
        for column in group:
            i = columns.index(column)
            with col1 if i < 5 else col2 if i < 10 else col3:
                if dtype[column] == "object":
                    input_data[column] = st.text_input(f"Please insert: {column}")
                elif dtype[column] == "int64":
                    input_data[column] = st.number_input(f"Please insert: {column}")
                else:
                    input_data[column] = st.number_input(f"Please insert: {column}")

    model_selection = st.selectbox("Which model would you like to predict the price?",
                                   ("Linear Regression", "Lasso Regression", "Random Forest Regressor", "Gradient Boosting Regressor",
                                    "AdaBoost Regressor", "Neural Network"), key="model_selection")
    print(input_data)
    if st.button("Show Car Price Result"):
        price_result = auto_price_predicition(input_data, model_selection)
        if model_selection != "Neural Network": 
            st.markdown(f"### :green[Der geschätze Preis beträgt: {str(round(price_result[0],2))} $]")
        else:
            st.markdown(f"### :green[Der geschätze Preis beträgt: {str(round(price_result[0][0], 2))} $]")


def main():
    st.set_page_config(layout="wide")
    with st.sidebar:
        selected = option_menu("Navigation", ["Hauptseite", 'Analyse des Datensatzes', "Vorhersage des Auto Preis"], 
            icons=['house', 'bar-chart-line-fill', "graph-up-arrow"], menu_icon="list", default_index=0)
    
    if (selected == "Hauptseite"):
        st.header("Data Science Projekt")
        st.text("Willkommen auf meiner Projekt-Seite, auf der ich mein Data Science-Projekt zur Analyse und Vorhersage des Verkaufspreises von Autos vorstelle.")
        st.subheader("Über das Projekt")
        st.text("""In diesem Projekt habe ich einen umfangreichen Datensatz von Autos analysiert, um wertvolle Einblicke in die Faktoren zu gewinnen, die den Verkaufspreis von Autos beeinflussen. Das Ziel dieses 
Projekts war zweigeteilt:
    1. Deskriptive Analyse: Zuerst habe ich eine umfassende deskriptive Analyse der vorhandenen Daten durchgeführt. Dabei habe ich die wichtigsten Merkmale der Autos untersucht,
         statistische Trends identifiziert und visuelle Darstellungen erstellt, um die Daten besser zu verstehen.
    2. Vorhersage des Verkaufspreises: Anschließend habe ich eine maschinelle Lernmodellierung implementiert, um den Verkaufspreis von Autos vorherzusagen. Dabei habe ich verschiedene
         Regressionstechniken angewendet und die besten Modelle ausgewählt, um genaue Preisprognosen zu erstellen.""")
        st.subheader("Herausforderungen und Lösungen")
        st.text("""Während des Projekts stand ich vor verschiedenen Herausforderungen, wie fehlende Daten, Datenbereinigung und die Auswahl der optimalen Merkmale. Ich habe diese Herausforderungen gemeistert,
indem ich sorgfältige Datenbereinigung und Feature-Engineering-Techniken angewendet habe.""")
        st.subheader("Verwendete Technologien")
        st.text("""Ich habe verschiedene Technologien und Tools in diesem Projekt eingesetzt, darunter:
    - Python
    - Jupyter Notebook
    - Pandas
    - NumPy
    - Matplotlib und Seaborn für Visualisierungen
    - Scikit-Learn für maschinelles Lernen
    - Keras und Tensorflow für Nueronale Netze""")
        st.subheader("Ergebnisse und Auswirkungen")
        st.text("""Durch dieses Projekt konnte ich wertvolle Erkenntnisse über den Automarkt gewinnen und genaue Vorhersagen des Verkaufspreises von Autos ermöglichen. Dies kann für Autohändler und Käufer
gleichermaßen von Nutzen sein, um fundierte Entscheidungen zu treffen.""")

        
        
        
    if (selected == "Analyse des Datensatzes"):
        st.header("Analyse des Datensatzes")
        st.subheader("Datenquelle")
        st.text("""Die Daten stammen aus einer umfangreichen Sammlung von Automobilinformationen. Zu finden ist der Datensatz unter folgenden URL:
https://figshare.com/articles/figure/DVM-CAR_Dataset/19586296/2""")
        st.subheader("Datenstruktur")
        st.text("""Unsere Daten umfassen Informationen zu verschiedenen Automarken, Modellen und technischen Spezifikationen, darunter:
    - Hersteller (Maker)
    - Modell (Genmodel)
    - Baujahr (Reg_Year)
    - Laufleistung (Runned_miles)
    - Motorleistung (Engine_power)
    - Abmessungen (Width, Length)
    - Durchschnittlicher Kraftstoffverbrauch (Average MPG)
    - Sitzplätze (Seat number)
    - Anzahl der Türen (Door number)
    - Karosserietyp (Bodytype)
    - Getriebetyp (Gearbox)
    - Kraftstoffart (Fuel type)
    - Farbe (Color)""")
        
        ## Erste Visualisierung
        st.subheader("Deskriptive Datenanalyse")
        fig1, ax01 = plt.subplots(figsize=(6,2))
        (df_begin.isna().sum()/df_begin.shape[0]).sort_values(ascending=False).plot(kind="bar")
        plt.title("Prozentualer Anteil an Null-Werten der verschiedenen Attribute des Datensatzes", fontsize=7)
        plt.xlabel("Attribute Datensatz", fontsize=6)
        plt.ylabel("Prozent", fontsize=6)
        ax01.tick_params(axis='both', labelsize=5)
        st.pyplot(fig1, use_container_width=False)
        
        ## Zweite Visualisierung
        fig2, ax02 = plt.subplots(figsize=(6,3))
        sns.heatmap(df_begin.corr(), annot=True, annot_kws={"fontsize":6}, cbar=False)
        plt.title("Korrelations-Heatmap", fontsize=7)
        ax02.tick_params(axis='both', labelsize=5)
        st.pyplot(fig2, use_container_width=False)
        
        ## Dritte Visualisierung
        st.markdown("#### Verteilung aller numerischer Datenwerte")
        fig3, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(10,7))
        axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
        fig3.suptitle('Verteilung aller numerischen Datenwerte')
        for col, ax in zip(num_cols, axes):
            ax.hist(df[col], bins=100)
            ax.set_title(col, fontsize=5)
            ax.tick_params(axis='both', labelsize=5)
        st.pyplot(fig3, use_container_width=False)
        
        ##Vierte Visualisierung
        fig4, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(12,12))
        axes2 = [ax1, ax2, ax3, ax4]
        for cat, ax in zip(cat_cols[2:], axes2):
            sns.countplot(data = df,x= cat, ax=ax)
            ax.tick_params(axis='x', labelrotation=60, labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
            ax.set_ylabel(ax.get_ylabel(), fontsize=6)
        st.pyplot(fig4, use_container_width=False)
        
        ##Fünfte Visualisierung  
        st.markdown("#### Anzahl Autos pro Hersteller im Datensatz:")
        fig5, ax11 = plt.subplots(figsize=(6,2))
        df.maker.value_counts().plot(kind="bar")
        plt.title("Anzahl Autos pro Hersteller im Datensatz", fontsize=7)
        plt.xlabel("Hersteller", fontsize=6)
        plt.xticks(fontsize=6)
        plt.ylabel("Anzahl Autos", fontsize=6)
        plt.yticks(fontsize=6)
        st.pyplot(fig5, use_container_width=False)
        
        ##Sechste Visualisierung      
        st.markdown("#### Durchschnittlicher Preis je Hersteller")
        mean_maker = df.groupby(by=["maker"])["price"].mean().sort_values(ascending=False)
        fig6, ax12 = plt.subplots(figsize=(6,2))
        (mean_maker[mean_maker < 1000000]).plot(kind="bar")
        plt.title("Durchschnittlicher Preis je Hersteller(Auswahl)", fontsize=7)
        plt.xlabel("Hersteller", fontsize=7)
        plt.xticks(fontsize=6)
        plt.ylabel("Preis", fontsize=7)
        plt.yticks(fontsize=6)
        st.pyplot(fig6, use_container_width=False) 
          
        ##Siebte Visualisierung
        df_maker = df[df["maker"].isin(list(df["maker"].value_counts().head().index))]
        df_maker = df_maker[df_maker["bodytype"].isin(["Saloon", "Convertible", "SUV", "Estate", "Coupe", "Hatchback"])]
        plot_test = df_maker.groupby(by=["reg_year", "maker"])["price"].mean()
        plot_test_df = pd.DataFrame(plot_test).reset_index()
        plot_test_df = plot_test_df[plot_test_df.reg_year > 1990]
        fig7, ax13 = plt.subplots(figsize=(6,4))
        sns.lineplot(x=plot_test_df["reg_year"], y=plot_test_df["price"], hue=plot_test_df["maker"])
        plt.legend(fontsize=6)
        plt.xticks(fontsize=6)
        plt.xlabel("Zulassungsjahr", fontsize=6)
        plt.ylabel("Preis", fontsize=6)
        plt.yticks(fontsize=6)
        plt.title("Durchschnittlicher Preis der größten Hersteller über Zeit", fontsize=7)
        st.pyplot(fig7, use_container_width=False)
        
        ## Achte Visualisierung
        scatter_df = df.groupby(by=["maker", "genmodel"])[["runned_miles", "price"]].mean().reset_index()
        filtered = scatter_df[(scatter_df["price"] < 100000) & (scatter_df["runned_miles"] < 400000) & (scatter_df["maker"].isin(scatter_df["maker"].value_counts().head().index))]     
        fig8, ax14 = plt.subplots(figsize=(6,4))
        sns.scatterplot(data=filtered, y="price", x="runned_miles", hue="maker")
        plt.title("Verhältnis Laufleistung und Preis der größten Hersteller", fontsize=7)
        plt.xlabel("Laufleistung", fontsize=6)
        plt.ylabel("Preis", fontsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.legend(fontsize=6)
        st.pyplot(fig8, use_container_width=False)
        
    if (selected == "Vorhersage des Auto Preis"):
        st.title("Vorhersage des Auto Preis")
        st.text("Bitte geben Sie in die unteren Eingabefelder die Daten ihres Fahrzeuges ein.")
        st.text("Beachten Sie dabei die Groß- und Kleinschreibung und dass die Sprache der Eingabewerte Englisch ist.")
        
        prediction()

if __name__ == "__main__":
    main()