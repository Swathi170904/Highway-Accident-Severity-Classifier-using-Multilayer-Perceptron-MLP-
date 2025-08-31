import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

st.set_page_config(page_title="Highway Accident Severity Classifier", layout="wide")


page = st.sidebar.radio("Navigation", [
    "Home", "Data", "EDA", "Models", "Prediction"
])
def home_page():
    st.title(" Highway Accident Severity Classifier")
    st.write("Upload your traffic accident CSV file to begin.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="home_upload")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state['df'] = df
        st.success("File uploaded successfully!")
        st.write(" Dataset Preview")
        st.dataframe(df.head())
    else:
        st.info("Please upload a CSV file to continue.")


def data_page():
    st.title(" Data Inspection & Cleaning")
    df = st.session_state.get('df')
    if df is None:
        st.warning("Please upload a dataset in the Home page first.")
        return
    st.write(" Summary Statistics")
    st.dataframe(df.describe())
    st.write(" Missing Values")
    st.write(df.isnull().sum())
    st.write(" First 5 Rows")
    st.dataframe(df.head())
    st.session_state['df_clean'] = df.copy()


def eda_page():
    st.title(" Exploratory Data Analysis (EDA)")
    df = st.session_state.get('df_clean')
    if df is None:
        st.warning("Please clean your data in the Data page first.")
        return
    st.write("### Correlation Heatmap")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12, 8))
    if not numeric_df.empty:
        heatmap = sns.heatmap(
            numeric_df.corr(), annot=True, cmap='mako', center=0, ax=ax,
            annot_kws={"size": 10}, fmt=".2f"
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=18, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(rotation=0, fontsize=12)
        st.pyplot(fig)
    else:
        st.info("No numeric columns available for correlation heatmap.")
    st.write("### Pairplot (first 5 numeric features)")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] >= 2:
        pairplot_fig = sns.pairplot(numeric_df.iloc[:, :5])
        st.pyplot(pairplot_fig)
    else:
        st.info("Not enough numeric columns for pairplot. At least two are required.")
def models_page():
    st.title(" Model Training & Evaluation")
    df = st.session_state.get('df_clean') 
    if df is None:
        st.warning("Please clean your data in the Data page first.")
        return
    if 'most_severe_injury' not in df.columns:
        st.error("Dataset must contain a 'most_severe_injury' column as target.")
        return
    X = df.drop(columns=['most_severe_injury'])
    y = df['most_severe_injury']
    max_rows = 2000
    if len(X) > max_rows:
        st.info(f"Dataset has {len(X)} rows. Using a random sample of {max_rows} rows for faster model training.")
        sample_idx = np.random.choice(X.index, size=max_rows, replace=False)
        X = X.loc[sample_idx]
        y = y.loc[sample_idx]
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_le = LabelEncoder()
    y_train = y_le.fit_transform(y_train)
    y_test = y_le.transform(y_test)
    if len(np.unique(y_train)) < 2:
        st.warning("Model training skipped: Only one class present in the target column after cleaning. Please check your data or adjust cleaning settings.")
        return
    models = {
        'Logistic': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(n_estimators=50),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True, max_iter=500),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50),
    }
    results = {}
    with st.spinner('Training models, please wait...'): 
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = acc
    best_model = max(results, key=results.get)
    st.write("### Model Accuracies")
    st.dataframe(pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy']))
    st.markdown("---")
    st.subheader("Best Model Summary & Visualization")
    st.success(f"Best Model: {best_model} (Accuracy: {results[best_model]:.2f})")
    st.write(f"Reason: {best_model} performed best based on accuracy compared to other models.")
    from sklearn.metrics import ConfusionMatrixDisplay
    best_model_instance = models[best_model]
    y_pred_best = best_model_instance.predict(X_test)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_best, ax=ax_cm, cmap='Blues', colorbar=False)
    ax_cm.set_title(f"Confusion Matrix: {best_model}")
    st.pyplot(fig_cm)
def prediction_page():
    st.title("Predict Severity with MLP")
    df = st.session_state.get('df_clean')
    if df is None:
        st.warning("Please clean your data in the Data page first.")
        return
    if 'most_severe_injury' not in df.columns:
        st.error("Dataset must contain a 'most_severe_injury' column as target.")
        return
    X = df.drop(columns=['most_severe_injury'])
    y = df['most_severe_injury']
    
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    y_le = LabelEncoder()
    y_train_enc = y_le.fit_transform(y_train)
    y_test_enc = y_le.transform(y_test)
    
    max_rows = 2000
    if len(X_train) > max_rows:
        sample_idx = np.random.choice(np.arange(len(X_train)), size=max_rows, replace=False)
        X_train_sample = X_train[sample_idx]
        y_train_enc_sample = y_train_enc[sample_idx]
    else:
        X_train_sample = X_train
        y_train_enc_sample = y_train_enc
   
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(np.unique(y_train_enc)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_sample, y_train_enc_sample, epochs=5, batch_size=32, verbose=0)
    
    from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
    y_pred_test = np.argmax(model.predict(X_test), axis=1)
    mlp_acc = accuracy_score(y_test_enc, y_pred_test)
    st.info(f"MLP Test Accuracy: {mlp_acc:.2f}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(y_test_enc, y_pred_test, display_labels=y_le.classes_, ax=ax, cmap='Blues', colorbar=False)
    ax.set_title("MLP Confusion Matrix")
    st.pyplot(fig)
    st.write("### Enter new accident details for prediction:")
    input_data = {}
    for col in X.columns:
        if str(X[col].dtype) == 'object':
            input_data[col] = st.text_input(f"{col}")
        else:
            input_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))
    if st.button("Predict Severity"):
        input_df = pd.DataFrame([input_data])
        for col in input_df.columns:
            if str(X[col].dtype) == 'object':
                le = LabelEncoder()
                le.fit(list(X[col].astype(str)))
                input_df[col] = le.transform(input_df[col].astype(str))
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)
        pred_idx = np.argmax(pred)
        pred_class = y_le.inverse_transform([pred_idx])[0]
        pred_probs = pred[0]
        
        severe_labels = ["FATAL", "INCAPACITATING INJURY", "NONINCAPACITATING INJURY", "SERIOUS"]
        if pred_class.upper() in severe_labels or "SERIOUS" in pred_class.upper():
            binary_severity = "Severe"
        else:
            binary_severity = "Not Severe"
        st.success(f"Predicted Severity: {pred_class}  |  Binary: {binary_severity}")
        prob_df = pd.DataFrame({
            'Class': y_le.classes_,
            'Probability': pred_probs
        })
        st.write("#### Class Probabilities:")
        st.dataframe(prob_df.sort_values('Probability', ascending=False).reset_index(drop=True))

if page == "Home":
    home_page()
elif page == "Data":
    data_page()
elif page == "EDA":
    eda_page()
elif page == "Models":
    models_page()
elif page == "Prediction":
    prediction_page()