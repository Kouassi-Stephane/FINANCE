# 1. Installation des packages nécessaires
# pip install pandas numpy scikit-learn matplotlib seaborn streamlit ydata-profiling requests imbalanced-learn

# 2. Importation des données et exploration de base :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight

st.markdown('<h1 style="color:blue;">Prédiction compte bancaire</h1>', unsafe_allow_html=True)

# Charger les données (avec mise en cache)
@st.cache_data
def load_data():
    df = pd.read_csv("Financial_inclusion_dataset.csv")
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    return df

df = load_data()

# Affichage d’informations générales sur le jeu de données
st.write("Aperçu des données :")
st.dataframe(df.head())
st.write("Informations sur les données :")
st.dataframe(df.info())
st.write("Statistiques descriptives :")
st.dataframe(df.describe())

# 3. Création des rapports de profilage pandas :
# Générer le rapport de profilage
profile = ProfileReport(df, title="Rapport de profilage")
profile.to_file("output.html")

# Afficher le rapport dans Streamlit (nécessite une conversion en HTML)
st.write("Rapport de profilage (extrait) :")
st.components.v1.html(profile.to_html(), height=1000, scrolling=True)

# 4. Gérer les valeurs manquantes et corrompues :
st.write("Valeurs manquantes :")
st.dataframe(df.isnull().sum())

# Prétraitement des données (avec mise en cache)
@st.cache_data
def preprocess_data(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    df.drop_duplicates(inplace=True)
    df = pd.get_dummies(df, columns=['country', 'location_type', 'cellphone_access', 'gender_of_respondent',
                                    'relationship_with_head', 'marital_status', 'education_level', 'job_type'], drop_first=True)
    df['bank_account'] = df['bank_account'].map({'Yes': 1, 'No': 0})
    df.drop(['uniqueid', 'year'], axis=1, inplace=True)
    return df

df = preprocess_data(df)

# 6. Gérer les valeurs aberrantes (Visualisations et potentielle suppression)
if st.checkbox("Afficher les boxplots"):
    st.title("Distribution de l'âge")
    fig, ax = plt.subplots()
    sns.boxplot(x=df['age_of_respondent'], ax=ax)
    ax.set_title("Boîte à moustaches de l'âge des répondants")
    ax.set_xlabel("Âge")
    st.pyplot(fig)
    plt.close(fig)

    st.title("Distribution de la taille du ménage")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=df['household_size'], ax=ax)
    ax.set_title("Boîte à moustaches de la taille du ménage", fontsize=16)
    ax.set_xlabel("Taille du ménage", fontsize=14)
    st.pyplot(fig)
    plt.close(fig)

# 8. Entraîner et tester un classifieur de machine learning :
@st.cache_resource
def train_model(df, method='SMOTE'):
    X = df.drop('bank_account', axis=1)
    y = df['bank_account']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if method == 'SMOTE':
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train_resampled, y_train_resampled)
    elif method == 'class_weight':
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))
        model = RandomForestClassifier(random_state=42, class_weight=class_weights_dict)
        model.fit(X_train, y_train)
    else:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

    return model, X_test, y_test

method = st.selectbox("Méthode de gestion du déséquilibre :", ('Aucune', 'SMOTE', 'class_weight'))
model, X_test, y_test = train_model(df, method)

y_pred = model.predict(X_test)
st.write("Rapport de classification :")
st.text(classification_report(y_test, y_pred))
st.write(f"Précision du modèle : {accuracy_score(y_test, y_pred)}")

pickle.dump(model, open('model.pkl', 'wb'))

# 9. Créer une application streamlit et ajouter des champs de saisie :
st.title("Prédiction d'ouverture de compte bancaire")

def user_input_features():
    # ... (Votre code pour les entrées utilisateur : selectbox, slider, etc.)
    location_type = st.selectbox('Type de localisation',('Rural','Urban'))
    cellphone_access = st.selectbox('Accès au téléphone portable',('Yes','No'))
    household_size = st.slider('Taille du ménage', 1, 20, 1)
    age_of_respondent = st.slider('Âge du répondant', 16,100,1)
    gender_of_respondent = st.selectbox('Genre du répondant',('Male','Female'))
    job_type = st.selectbox('Type d\'emploi',('Farming and Fishing','Self employed','Formally employed Government','Formally employed Private','Informally employed','Remittance Dependent','Government Dependent','Other Income','No Income','Dont Know/Refuse to answer'))
    education_level = st.selectbox('Niveau d\'éducation',('No formal education','Primary education','Secondary education','Vocational/Specialised training','Tertiary education','Other/Dont know/RTA'))
    marital_status = st.selectbox('Statut marital',('Married/Living together','Divorced/Seperated','Widowed','Single/Never Married','Don’t know'))
    relationship_with_head = st.selectbox('Relation avec le chef de ménage',('Head of Household','Spouse','Child','Parent','Other relative','Other non-relatives','Dont know'))
    country = st.selectbox('Pays',('Kenya','Rwanda','Tanzania','Uganda'))
    data = {'location_type': location_type,
            'cellphone_access': cellphone_access,
            'household_size': household_size,
            'age_of_respondent': age_of_respondent,
            'gender_of_respondent': gender_of_respondent,
            'job_type':job_type,
            'education_level':education_level,
            'marital_status':marital_status,
            'relationship_with_head':relationship_with_head,
            'country':country
            }
    features = pd.DataFrame(data, index=[0])
    return features

df_user = user_input_features()
st.subheader('Paramètres entrés par l\'utilisateur')
st.write(df_user)

# 10. Importer le modèle ML et faire des prédictions (CODE CORRIGÉ ET COMPLÉTÉ) :
def prepare_user_data(df_user, df_train):
    # Concaténation pour assurer les mêmes colonnes après le one-hot encoding
    df_final = pd.concat([df_user, df_train.drop('bank_account', axis=1)], axis=0)

    # Encodage one-hot
    df_final = pd.get_dummies(df_final, columns=['location_type', 'cellphone_access', 'gender_of_respondent',
                                                'relationship_with_head', 'marital_status', 'education_level', 'job_type','country'], drop_first=True)

    # Sélection de la ligne utilisateur (la première après la concaténation)
    df_final = df_final[:1]

    # Alignement des colonnes avec celles du modèle entraîné (TRÈS IMPORTANT)
    train_cols = df_train.drop('bank_account', axis=1).columns
    for col in train_cols:
        if col not in df_final.columns:
            df_final[col] = 0 # Ajout de colonnes manquantes avec la valeur 0
    df_final = df_final[train_cols] # Réordonnancement des colonnes

    return df_final

df_final = prepare_user_data(df_user, df)

if st.button("Prédire"):
    prediction = model.predict(df_final)
    st.write(f"Prédiction : {'Oui' if prediction[0] == 1 else 'Non'}")
