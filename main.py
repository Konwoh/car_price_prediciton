import pickle
import pandas as pd
import streamlit as st
#from data_fitting import prediciton_preprocessing
from streamlit_option_menu import option_menu
from nn_model import nn_model
from PIL import Image

#laden der trainerten Modelle
loaded_model_lr = pickle.load(open("models/lr_model.pkl", "rb"))
loaded_model_lasso = pickle.load(open("models/lasso_model.pkl", "rb"))
loaded_column_transf = pickle.load(open("transformer_models/column_transf.pkl", "rb"))
#loaded_model_nn = keras.models.load_model("models/nn_model.keras")
loaded_model_rrf = pickle.load(open("models/rrf.pkl", "rb"))
loaded_rrf_pca = pickle.load(open("models/rrf_pca_model.pkl", "rb"))
loaded_pca = pickle.load(open("transformer_models/pca_transformer.pkl", "rb"))
loaded_gbr = pickle.load(open("models/gbr_model.pkl", "rb"))
loaded_abr = pickle.load(open("models/abr_model.pkl", "rb"))
columns = ["reg_year", "runned_miles", "engine_power", "width", "length", "average_mpg", "seat_num", "door_num", "maker", "genmodel", "color", "bodytype", "gearbox", "fuel_type"]

def load_my_model():
    model = nn_model()
    model.build((None, 683))
    model.load_weights('models/model_weights.h5')
    return model
loaded_model_nn = load_my_model()
def auto_price_predicition(input_data, model):
    df_test = pd.DataFrame(list(input_data.values())).T
    df_test.columns = columns
    input_pre = loaded_column_transf.transform(df_test)
    all_column_names = list(loaded_column_transf.named_transformers_['scaler'].get_feature_names_out()) \
    + list(loaded_column_transf.named_transformers_['encoder'].get_feature_names_out()) + ["reg_year"]
    df_prediction2 = pd.DataFrame(input_pre.toarray(), columns=all_column_names)
    input_pre_pca = loaded_pca.transform(df_prediction2)
    if model == "Linear Regression":
        return loaded_model_lr.predict(input_pre)
    elif model == "Lasso Regression":
        return loaded_model_lasso.predict(input_pre)
    elif model == "Neural Network":
        return loaded_model_nn.predict(input_pre)
    elif model == "Random Forest Regressor":
        return loaded_model_rrf.predict(input_pre)
    elif model == "Random Forest Regressor mit PCA":
        return loaded_rrf_pca.predict(input_pre_pca)
    elif model == "Gradient Boosting Regressor":
        return loaded_gbr.predict(input_pre)
    elif model == "AdaBoost Regressor":
        return loaded_abr.predict(input_pre)
    
def prediction():
    input_data = {}
    
    column_groups = [columns[:5], columns[5:10], columns[10:]]
    col1, col2, col3 = st.columns(3)

    for group in column_groups:
        for column in group:
            i = columns.index(column)
            with col1 if i < 5 else col2 if i < 10 else col3:
                if i < 8:
                    input_data[column] = st.number_input(f"Please insert: {column}") 
                else:
                    input_data[column] = st.text_input(f"Please insert: {column}")    
                         
    model_selection = st.selectbox("Which model would you like to predict the price?",
                                   ("Linear Regression", "Lasso Regression", "Random Forest Regressor", "Gradient Boosting Regressor",
                                    "AdaBoost Regressor", "Random Forest Regressor mit PCA", "Neural Network"), key="model_selection")
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
        st.write("""In diesem Projekt habe ich einen umfangreichen Datensatz von Autos analysiert, um wertvolle Einblicke in die Faktoren zu gewinnen, die den Verkaufspreis von Autos beeinflussen. Das Ziel dieses 
Projekts war zweigeteilt: \n 1. Deskriptive Analyse: Zuerst habe ich eine umfassende deskriptive Analyse der vorhandenen Daten durchgeführt. Dabei habe ich die wichtigsten Merkmale der Autos untersucht,
         statistische Trends identifiziert und visuelle Darstellungen erstellt, um die Daten besser zu verstehen. \n 2. Vorhersage des Verkaufspreises: Anschließend habe ich eine maschinelle Lernmodellierung implementiert, um den Verkaufspreis von Autos vorherzusagen. Dabei habe ich verschiedene
         Regressionstechniken angewendet und die besten Modelle ausgewählt, um genaue Preisprognosen zu erstellen.""")
        st.subheader("Herausforderungen und Lösungen")
        st.write("""Während des Projekts stand ich vor verschiedenen Herausforderungen, wie fehlende Daten, Datenbereinigung und die Auswahl der optimalen Merkmale. Ich habe diese Herausforderungen gemeistert,
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
        st.write("""Durch dieses Projekt konnte ich wertvolle Erkenntnisse über den Automarkt gewinnen und genaue Vorhersagen des Verkaufspreises von Autos ermöglichen. Dies kann für Autohändler und Käufer
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
        st.subheader("Explorative Datenanalyse")
        st.write("""Im folgenden werden einige Visualisierungen eingeblendet, die dem Lesenden ein besseres Verständnis für den Datensatz geben soll.""")
        st.subheader("Prozentualer Anteil fehlender Werte")
        image1 = Image.open('Visualizations/missing_values.png')
        st.image(image1, width=600)
        st.write("""Diese Grafik bietet einen Überblick über die Qualität unseres Datensatzes, indem sie den prozentualen Anteil fehlender Werte in verschiedenen 
                 Attributen darstellt. Ein hoher Prozentsatz von fehlenden Werten in einem Attribut kann auf Datenqualitätsprobleme hinweisen und beeinflusst die 
                 Analyse und Modellierung. In der Grafik ist deutlich zu sehen, dass die meisten Null-Values in den Attributen Annual_Tax, Top_Speed und Average_mpg
                 auftreten. Mögliche Lösungsansätze sind: \n 1. Löschen der jeweiligen Spalte \n 2. Ersetzen der Null-Values durch z.B. den Durchschnitt oder Median
                 Spalte""")
        st.subheader("Korrelations-Heatmap")
        image2 = Image.open('Visualizations/Korr_heatmap.png')
        st.image(image2, width=600)
        st.write("""Diese beeindruckende Heatmap bietet einen visuellen Einblick in die Korrelation zwischen den numerischen Attributen unseres Datensatzes. Die 
                 Heatmap ist ein leistungsstarkes Werkzeug, um Muster und Beziehungen zwischen verschiedenen Merkmalen zu erkennen. Zu beachten ist allerdings, dass
                 diese Heatmap auf den unbearbeiteten Datensatz angewendet wurde, wodurch manche numerische Spalten fehlen, da sie in dieser Form des Datensatzes noch
                 als Objekte codiert waren. Dennoch kann sehr gut die Stärke der linearen Beziehungen zwischen Preis und den restlichen Spalten beobachtet werden. Bei
                 den zugrundeliegenden Datensatz ist die lineare Beziehung zwischen Preis und Motorleistung am stärksten.""")
        st.subheader("Anzahl Autos pro Hersteller")
        image3 = Image.open('Visualizations/anzahl_autos.png')
        st.image(image3, width=750)
        st.write("""Diese Balkendiagramm-Visualisierung bietet einen Überblick über die Anzahl der Autos, die von verschiedenen Herstellern in unserem Datensatz
                 vertreten sind. Dies ermöglicht es, die Verteilung der Automarken im Datensatz zu verstehen. Die Hersteller die die meisten Autos in unseren
                 Datensatz haben, sind Ford, Audi, Vauxhall, Volkswagen und BMW""")
        st.subheader("Verteilungen aller numerischen Datensatz-Attribute")
        image4 = Image.open('Visualizations/num_verteilung.png')
        st.image(image4, width=900)
        st.write("""Diese Visualisierung zeigt die Verteilung der Werte in unseren numerischen Attributen. Die Darstellung erfolgt in einem Raster von Histogrammen,
                 wobei jedes Histogramm ein anderes numerisches Attribut darstellt. Besonders aufällig ist die Verteilung des Preis-Attributs. Da es in dem Datensatz
                 extrem teure Autos gibt, wird diese Verteilung als ein einzelner Strich dargestellt.""")
        st.subheader("Verteilungen aller kategorischen Datensatz-Attribute")
        image5 = Image.open('Visualizations/cat_verteilung.png')
        st.image(image5, width=750)
        st.write("""Diese Visualisierung zeigt die Verteilung der Werte in unseren kategorischen Attributen. Die Darstellung erfolgt in einem Raster von Histogrammen,
         wobei jedes Histogramm ein anderes kategorisches Attribut darstellt.""")
        st.subheader("Durchschnittlicher Preis je Hersteller")
        image6 = Image.open('Visualizations/mean_price.png')
        st.image(image6, width=800)
        st.write("""Diese Balkendiagramm-Visualisierung zeigt den Durchschnittspreis der Autos für ausgewählte Hersteller in unserem Datensatz. Sie bietet Einblicke 
                 in die Preisgestaltung unterschiedlicher Hersteller und kann bei der Analyse von Markenpräferenzen und Preisstrukturen nützlich sein. Zu sehen ist,
                 dass Lamborghini, McLaren, Ferrari und Rolls-Royce die mit Abstand teuersten Autos im Datensatz besitzen. Allerdings wurden in der Darstellung
                 nur Autos unter einem Preis von 1 Million $ berücksichtigt, da sonst die Ansehnlichkeit der Visualisierung darunter gelitten hätte.""")
        st.subheader("Durchschnittlicher Preis über Zeit")
        image7 = Image.open('Visualizations/price_over_time.png')
        st.image(image7, width=700)
        st.write("""Diese Liniendiagramm-Visualisierung veranschaulicht die Entwicklung des durchschnittlichen Auto-Preises im Laufe der Zeit für ausgewählte 
                 Hersteller. Sie ermöglicht es, Trends in der Preisgestaltung verschiedener Hersteller im Zeitverlauf zu identifizieren.""")
        st.subheader("Scatterplot von Laufleistung und Preis")
        image8 = Image.open('Visualizations/miles_price_scatter.png')
        st.image(image8, width=700)
        st.write("""Diese Streudiagramm-Visualisierung zeigt das Verhältnis zwischen der Laufleistung und dem Preis von Autos für ausgewählte Hersteller. Sie hilft
                 dabei, mögliche Zusammenhänge zwischen diesen beiden Faktoren zu erkennen und ihre Auswirkungen auf den Fahrzeugmarkt zu verstehen.""")
        
    if (selected == "Vorhersage des Auto Preis"):
        st.title("Vorhersage des Auto Preis")
        st.text("Bitte geben Sie in die unteren Eingabefelder die Daten ihres Fahrzeuges ein.")
        st.text("Beachten Sie dabei die Groß- und Kleinschreibung und dass die Sprache der Eingabewerte Englisch ist.")
        st.subheader("R2-Scores der verschiedenen Modelle")
        st.write("""Lineare Regression: -1.0700188322042872e+16 
                 \n Lasso Regression: 0.5988089083316444 
                 \n Random Forest Regressor: 0.861095177381202 
                 \n Gradient Boosting Regressor 0.8141688072472406 
                 \n AdaBoost Regressor: 0.61401889818197""")
        st.subheader("Vorhersage")
        prediction()

if __name__ == "__main__":
    main()