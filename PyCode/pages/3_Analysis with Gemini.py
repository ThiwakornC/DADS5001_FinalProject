import os
import pandas as pd
import plotly.express as px
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
###########################################################

st.set_page_config(
    page_title="Analysis with Gemini",
    page_icon="🤖",
)
st.sidebar.write("### 3_Analysis with Gemini")

@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\Thiwa\\dads5001\\Tools_Project\Game.csv", encoding='utf-8-sig') #chnage path
    return df
df = load_data()

# Machine Learning Mode
st.sidebar.write("### 3_Analysis with Gemini")
#########################################################
# Load .env
load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')

# Title of Streamlit app
st.title("Game Data Analysis")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a Game CSV file", type=["csv"])
###########################################################################
st.sidebar.write("### Analyze with Gemini")

# Option to use full dataset or a sample
use_full_data = st.sidebar.checkbox("Use Full Dataset", value=False)

# Select data for analysis
if use_full_data:
    selected_data = df
else:
    sample_size = st.sidebar.slider("Sample Size:", min_value=5, max_value=min(50, len(df)), value=20)
    selected_data = df.head(sample_size)

# Filter for safe columns
safe_columns = selected_data.select_dtypes(include=['float64', 'int64', 'object']).columns
selected_data = selected_data[safe_columns]

# Select columns for analysis
selected_columns = st.sidebar.multiselect(
    "Select Columns to Analyze:",
    options=selected_data.columns,
    default=list(selected_data.columns[:4])
)

# Ensure data is valid
if selected_data.empty or len(selected_columns) == 0:
    st.warning("Please select valid data and columns for analysis.")
else:
    # Filter selected columns
    filtered_data = selected_data[selected_columns]
    st.write("### Data Sent to Gemini")
    st.write(filtered_data)

    # Example commands
    st.write("**Example Commands:**")
    st.write("- Summarize the relationship between 'Average playtime forever' and 'Price'.")
    st.write("- Identify the top 5 genres with the highest average Metacritic score.")

    # User input for Gemini
    user_command = st.text_area(
        "Enter your command/query for Gemini:",
        "Summarize the data and highlight key insights."
    )

    # Analyze data with Gemini
    if st.button("Analyze with Gemini"):
        try:
            # Convert data to string
            data_text = filtered_data.to_string(index=False)
            # Send command to Gemini
            response = model.generate_content([f"{user_command}\n\nHere is the dataset:\n{data_text}"])
            # Display Gemini's response
            st.write("#### Gemini's Analysis")
            st.write(response.text)
        except Exception as e:
            st.error(f"An error occurred during Gemini analysis: {e}")
########################################################################################
