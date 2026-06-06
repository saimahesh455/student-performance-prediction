import streamlit as st
import numpy as np
import pandas as pd
import os

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Student Performance App", layout="wide")


st.title("🎓 Student Performance Prediction App")
st.markdown("### Predict Math Score using ML Models")


@st.cache_data
def load_data():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "StudentsPerformance.csv")
    return pd.read_csv(file_path)

try:
    df = load_data()
except:
    st.error("❌ CSV file not found. Make sure it's in the same folder as app.py")
    st.stop()

if st.checkbox("📂 Show Dataset"):
    st.dataframe(df)

# -----------------------------
# FEATURES
# -----------------------------
X = df.drop(columns=['math score'], axis=1)
y = df['math score']

# -----------------------------
# PREPROCESSING
# -----------------------------
num_cols = X.select_dtypes(exclude='object').columns
cat_cols = X.select_dtypes(include='object').columns

preprocessor = ColumnTransformer(
    [
        ("OneHotEncoder", OneHotEncoder(), cat_cols),
        ("StandardScaler", StandardScaler(), num_cols)
    ]
)

x = preprocessor.fit_transform(X)

# -----------------------------
# SPLIT DATA
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=23
)

# -----------------------------
# SIDEBAR MODEL SELECTION
# -----------------------------
st.sidebar.header("⚙️ Model Settings")

model_name = st.sidebar.selectbox(
    "Choose Model",
    ["Linear Regression", "Lasso", "Ridge", "KNN", "Decision Tree", "Random Forest"]
)

def get_model(name):
    if name == "Linear Regression":
        return LinearRegression()
    elif name == "Lasso":
        return Lasso()
    elif name == "Ridge":
        return Ridge()
    elif name == "KNN":
        return KNeighborsRegressor()
    elif name == "Decision Tree":
        return DecisionTreeRegressor()
    else:
        return RandomForestRegressor()

model = get_model(model_name)

# -----------------------------
# TRAIN MODEL
# -----------------------------
st.markdown("## 📊 Model Training")

if st.button("🚀 Train Model"):
    try:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.success("✅ Model Trained Successfully!")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("MSE", f"{mse:.2f}")
        col3.metric("RMSE", f"{rmse:.2f}")
        col4.metric("R2 Score", f"{r2:.2f}")

        st.progress(float(max(min(r2, 1), 0)))

        # -----------------------------
        # PLOT
        # -----------------------------
        st.markdown("## 📈 Actual vs Predicted")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

        sns.regplot(x=y_test, y=y_pred, ci=None, color='red', ax=ax)

        st.pyplot(fig)

        # -----------------------------
        # TABLE
        # -----------------------------
        pred_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Difference': y_test - y_pred
        })

        st.markdown("## 📋 Predictions Table")
        st.dataframe(pred_df)

        # DOWNLOAD
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", csv, "predictions.csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# -----------------------------
# MODEL COMPARISON
# -----------------------------
st.markdown("## 🏆 Compare All Models")

if st.checkbox("Show Comparison"):

    models = {
        'LR': LinearRegression(),
        'Lasso': Lasso(),
        'Ridge': Ridge(),
        'KNN': KNeighborsRegressor(),
        'DT': DecisionTreeRegressor(),
        'RF': RandomForestRegressor()
    }

    model_list = []
    r2_list = []

    for name, m in models.items():
        m.fit(x_train, y_train)
        y_pred = m.predict(x_test)
        r2 = r2_score(y_test, y_pred)

        model_list.append(name)
        r2_list.append(r2)

    results = pd.DataFrame({
        "Model": model_list,
        "R2 Score": r2_list
    }).sort_values(by="R2 Score", ascending=False)

    st.dataframe(results)

    best_model = results.iloc[0]
    st.success(f"🏆 Best Model: {best_model['Model']} (R2: {best_model['R2 Score']:.2f})")

# -----------------------------
# USER INPUT PREDICTION
# -----------------------------
st.sidebar.header("🎯 Predict Student Score")

gender = st.sidebar.selectbox("Gender", df['gender'].unique())
race = st.sidebar.selectbox("Race/Ethnicity", df['race/ethnicity'].unique())
parent_edu = st.sidebar.selectbox("Parental Education", df['parental level of education'].unique())
lunch = st.sidebar.selectbox("Lunch", df['lunch'].unique())
test_course = st.sidebar.selectbox("Test Prep Course", df['test preparation course'].unique())

reading = st.sidebar.slider("Reading Score", 0, 100, 50)
writing = st.sidebar.slider("Writing Score", 0, 100, 50)

if st.sidebar.button("🎯 Predict"):

    try:
        input_df = pd.DataFrame({
            'gender': [gender],
            'race/ethnicity': [race],
            'parental level of education': [parent_edu],
            'lunch': [lunch],
            'test preparation course': [test_course],
            'reading score': [reading],
            'writing score': [writing]
        })

        input_data = preprocessor.transform(input_df)

        # Ensure model is trained
        model.fit(x_train, y_train)

        prediction = model.predict(input_data)

        st.sidebar.success(f"📊 Predicted Math Score: {prediction[0]:.2f}")

    except Exception as e:
        st.sidebar.error(f"Error: {e}")