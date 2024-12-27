import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import streamlit as st
from imblearn.over_sampling import SMOTE
from io import StringIO
import pickle
import plotly.express as px
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="Inclusion Financière en Afrique", layout="wide")

# Titre principal et description
st.title("🌍 Inclusion Financière en Afrique")
st.markdown("""
Cette application analyse l'inclusion financière en Afrique de l'Est en utilisant un ensemble de données 
contenant des informations démographiques de 33 600 personnes. L'objectif est de prédire la probabilité 
qu'une personne possède ou utilise un compte bancaire.
""")

# Fonction pour charger les données
@st.cache_data
def load_data():
    df = pd.read_csv("Financial_inclusion_dataset.csv")
    return df

# Chargement des données
df = load_data()

# Création des onglets
tabs = st.tabs([
    "📊 Exploration des Données",
    "🧹 Prétraitement",
    "📈 Visualisations",
    "🤖 Modélisation",
    "🎯 Prédictions"
])

# Onglet 1: Exploration des Données
with tabs[0]:
    st.header("Exploration des Données")
    
    # Informations générales
    st.subheader("Aperçu des données")
    st.dataframe(df.head())
    
    # Statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe())
    
    # Informations sur les colonnes
    st.subheader("Informations sur les colonnes")
    buffer = StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    
    # Distribution des variables catégorielles
    st.subheader("Distribution des variables catégorielles")
    cat_cols = ['country', 'location_type', 'cellphone_access', 'gender_of_respondent',
                'relationship_with_head', 'marital_status', 'education_level', 'job_type']
    
    for col in cat_cols:
        col_count = df[col].value_counts().reset_index()
        col_count.columns = [col, 'count']  # Renommer les colonnes après reset_index()
        fig = px.bar(col_count,
                     x=col, y='count',
                     title=f'Distribution de {col}')
        st.plotly_chart(fig)

# Onglet 2: Prétraitement
with tabs[1]:
    st.header("Prétraitement des Données")
    
    # Affichage des valeurs manquantes
    st.subheader("Valeurs manquantes")
    missing_values = pd.DataFrame({
        'Column': df.columns,
        'Missing Values': df.isnull().sum(),
        'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(missing_values)
    
    # Doublons
    st.subheader("Doublons")
    duplicates = df.duplicated().sum()
    st.write(f"Nombre de doublons: {duplicates}")
    
    # Valeurs aberrantes
    st.subheader("Détection des valeurs aberrantes")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        fig = go.Figure()
        fig.add_trace(go.Box(y=df[col], name=col))
        fig.update_layout(title=f'Boîte à moustaches - {col}')
        st.plotly_chart(fig)

# Onglet 3: Visualisations
with tabs[2]:
    st.header("Visualisations")
    
    # Distribution de l'âge par pays
    fig = px.box(df, x='country', y='age_of_respondent',
                 title="Distribution de l'âge par pays")
    st.plotly_chart(fig)
    
    # Compte bancaire par niveau d'éducation
    fig = px.histogram(df, x='education_level', color='bank_account',
                      title="Possession de compte bancaire par niveau d'éducation")
    st.plotly_chart(fig)
    
    # Carte de chaleur des corrélations
    numeric_df = df.select_dtypes(include=[np.number])
    fig = px.imshow(numeric_df.corr(),
                    title="Carte de chaleur des corrélations")
    st.plotly_chart(fig)

# Onglet 4: Modélisation
with tabs[3]:
    st.header("Modélisation")
    
    # Fonction de prétraitement
    @st.cache_data
    def preprocess_data(df):
        df_processed = df.copy()
        
        # Encodage des variables catégorielles
        df_processed['bank_account'] = df_processed['bank_account'].map({'Yes': 1, 'No': 0})
        df_processed = pd.get_dummies(df_processed, columns=[
            'country', 'location_type', 'cellphone_access', 'gender_of_respondent',
            'relationship_with_head', 'marital_status', 'education_level', 'job_type'
        ])
        
        # Suppression des colonnes non nécessaires
        df_processed.drop(['uniqueid', 'year'], axis=1, inplace=True)
        
        return df_processed
    
    # Préparation des données
    df_processed = preprocess_data(df)
    X = df_processed.drop('bank_account', axis=1)
    y = df_processed['bank_account']
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # SMOTE pour gérer le déséquilibre
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    # Évaluation
    y_pred = model.predict(X_test)
    
    st.subheader("Rapport de classification")
    st.text(classification_report(y_test, y_pred))
    
    # Matrice de confusion
    st.subheader("Matrice de confusion")
    fig = px.imshow(confusion_matrix(y_test, y_pred),
                    labels=dict(x="Prédit", y="Réel"),
                    x=['Non', 'Oui'],
                    y=['Non', 'Oui'])
    st.plotly_chart(fig)
    
    # Sauvegarde du modèle
    pickle.dump(model, open('model.pkl', 'wb'))

# Onglet 5: Prédictions
with tabs[4]:
    st.header("Prédiction d'ouverture de compte bancaire")
    
    # Interface de saisie
    col1, col2 = st.columns(2)
    
    with col1:
        country = st.selectbox('Pays', ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
        age = st.slider('Âge', 16, 100, 30)
        gender = st.selectbox('Genre', ['Male', 'Female'])
        education = st.selectbox('Niveau d\'éducation', [
            'No formal education',
            'Primary education',
            'Secondary education',
            'Tertiary education',
            'Vocational/Specialised training'
        ])
    
    with col2:
        location = st.selectbox('Type de localisation', ['Rural', 'Urban'])
        household_size = st.slider('Taille du ménage', 1, 20, 4)
        phone = st.selectbox('Accès au téléphone', ['Yes', 'No'])
        job = st.selectbox('Type d\'emploi', [
            'Formally employed Private',
            'Self employed',
            'Farming and Fishing',
            'Informally employed',
            'Other'
        ])
    
    # Bouton de prédiction
    if st.button('Prédire'):
        # Création du DataFrame pour la prédiction
        input_data = pd.DataFrame({
            'country': [country],
            'age_of_respondent': [age],
            'gender_of_respondent': [gender],
            'location_type': [location],
            'household_size': [household_size],
            'cellphone_access': [phone],
            'education_level': [education],
            'job_type': [job],
            'relationship_with_head': ['Head of Household'],
            'marital_status': ['Single/Never Married']
        })
        
        # Prétraitement des données utilisateur
        user_processed = preprocess_data(pd.concat([input_data, df.iloc[:1]]).iloc[:1])
        user_processed = user_processed.reindex(columns=X.columns, fill_value=0)
        
        # Prédiction
        prediction = model.predict(user_processed)
        proba = model.predict_proba(user_processed)
        
        # Affichage des résultats
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Probabilité d'avoir un compte bancaire: {proba[0][1]:.2%}")
        with col2:
            st.info("Prédiction: " + ("Compte bancaire probable" if prediction[0] == 1 else "Compte bancaire peu probable"))
