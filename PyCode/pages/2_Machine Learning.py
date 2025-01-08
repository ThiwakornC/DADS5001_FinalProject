import os
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="Machine Learning",
    page_icon="📈",
)

@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\Thiwa\\dads5001\\Tools_Project\Game.csv", encoding='utf-8-sig') #change path
    return df
df = load_data()

# Machine Learning Mode
st.sidebar.write("### Machine Learning")

# Sub-options for Machine Learning
ml_mode = st.sidebar.radio(
    "Choose a Machine Learning task:",
    ("Feature Importance", "Regression")
)

st.write("### Machine Learning Task: ", ml_mode)
st.write('You can select target column on side bar.')

if ml_mode == "Feature Importance":
    # Select target column
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    target_column = st.sidebar.selectbox("Select Target Column:", options=numeric_columns)

    if target_column and 'Genres' in df.columns:
        # One-Hot Encoding for 'Genres'
        X = pd.get_dummies(df['Genres'].str.get_dummies(sep=','), drop_first=True)
        y = df[target_column]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Predict results
        y_pred = model.predict(X_test)

        # Display feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.write("### Feature Importance")
        st.write(feature_importance)
        fig = px.bar(feature_importance, x='Feature', y='Importance', title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("The dataset must contain a 'Genres' column and a selected target column.")

elif ml_mode == "Regression":
    if 'Year released' in df.columns and 'Genres' in df.columns:
        # Allow user to filter by year
        unique_years = sorted(df['Year released'].dropna().unique().astype(int))
        selected_years = st.sidebar.multiselect(
            "Select Years for Regression:",
            options=unique_years,
            default=unique_years
        )

        # Filter data by selected years
        if selected_years:
            filtered_df = df[df['Year released'].isin(selected_years)]

            # One-Hot Encoding for 'Genres'
            X = pd.get_dummies(filtered_df['Genres'].str.get_dummies(sep=','), drop_first=True)

            # Select target column
            numeric_columns = filtered_df.select_dtypes(include=['float64', 'int64']).columns
            target_column = st.sidebar.selectbox("Select Target Column for Regression:", options=numeric_columns)

            if target_column:
                y = filtered_df[target_column]

                # Split data into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train Linear Regression model
                lin_reg = LinearRegression()
                lin_reg.fit(X_train, y_train)

                # Predict results
                y_pred = lin_reg.predict(X_test)

                # Display regression metrics
                st.write("### Regression Metrics")
                
                # Display coefficients
                coefficients = pd.DataFrame({
                    'Feature': X.columns,
                    'Coefficient': lin_reg.coef_
                }).sort_values(by='Coefficient', ascending=False)

                st.write("### Regression Coefficients")
                st.write(coefficients)
                fig = px.bar(coefficients, x='Feature', y='Coefficient', title="Regression Coefficients")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select a valid target column for regression.")
        else:
            st.warning("Please select at least one year.")
    else:
        st.warning("The dataset must contain 'Year released' and 'Genres' columns for regression.")
