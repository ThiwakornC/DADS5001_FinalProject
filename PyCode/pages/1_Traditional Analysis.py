import os
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import pandas as pd
import base64
from PIL import Image
import requests
from io import BytesIO
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import re
from collections import Counter
import random

st.set_page_config(
    page_title="Traditional Analysis",
    page_icon="📊",
)

st.sidebar.write("### Traditional Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\Thiwa\\dads5001\\Tools_Project\Game.csv", encoding='utf-8-sig') #change path
    return df
df = load_data()

# Scatter Plot with flexible options
if st.sidebar.checkbox("Show Scatter Plot"):
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    x_axis = st.sidebar.selectbox("Select X-axis:", options=numeric_columns, index=0)
    y_axis = st.sidebar.selectbox("Select Y-axis:", options=numeric_columns, index=1)
    remove_outliers_option = st.sidebar.checkbox("Remove Outliers")
    if remove_outliers_option:
        filtered_df = remove_outliers(df, x_axis)
        filtered_df = remove_outliers(filtered_df, y_axis)
        st.write(f"Data points after removing outliers: {len(filtered_df)}")
    else:
        filtered_df = df
    scatter_fig = px.scatter(
        filtered_df,
        x=x_axis,
        y=y_axis,
        title=f'{x_axis} vs {y_axis}' +
                (" (Without Outliers)" if remove_outliers_option else ""),
        labels={x_axis: x_axis, y_axis: y_axis},
        hover_data=['Name','Genres'],
        template='plotly_white'
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

# Bar Chart with Top 10 Tags
if st.sidebar.checkbox("Show Top 10 Tags"):
    if 'Genres' in df.columns and 'Year released' in df.columns:
        # Allow user to select specific years
        unique_years = sorted(df['Year released'].dropna().unique().astype(int))
        selected_years = st.sidebar.multiselect(
            "Select Years:",
            options=unique_years,
            default=unique_years
        )

        # Filter data based on selected years
        if selected_years:
            filtered_df = df[df['Year released'].isin(selected_years)]

            # One-hot encoding for Genres
            tags_one_hot = filtered_df['Genres'].str.get_dummies(sep=',')

            # Create metrics for all numeric columns
            tag_metrics = {
                'Total Average playtime forever': tags_one_hot.multiply(filtered_df['Average playtime forever'], axis=0).sum(),
                'Total Price': tags_one_hot.multiply(filtered_df['Price'], axis=0).mean(),
                'Metacritic score': tags_one_hot.multiply(filtered_df['Metacritic score'], axis=0).sum(),
                'Recommendations': tags_one_hot.multiply(filtered_df['Recommendations'], axis=0).sum(),
                'Positive': tags_one_hot.multiply(filtered_df['Positive'], axis=0).sum(),
                'Negative': tags_one_hot.multiply(filtered_df['Negative'], axis=0).sum(),
                'Median playtime forever': tags_one_hot.multiply(filtered_df['Median playtime forever'], axis=0).sum(),
                'Peak CCU': tags_one_hot.multiply(filtered_df['Peak CCU'], axis=0).sum(),
                'Total number of reviews': tags_one_hot.multiply(filtered_df['Total number of reviews'], axis=0).sum(),
                'Estimated owners by cat': tags_one_hot.multiply(filtered_df['Estimated owners by cat'], axis=0).sum(),
                'Estimated owners by review': tags_one_hot.multiply(filtered_df['Estimated owners by review'], axis=0).sum(),
            }

            # Allow user to select metric (Y-axis)
            selected_metric = st.sidebar.selectbox(
                "Choose a metric for Y-axis:",
                list(tag_metrics.keys()),
                index=0
            )

            tag_values = tag_metrics[selected_metric].sort_values(ascending=False)

            st.write(f"### Top 10 Tags by {selected_metric} (Selected Years: {', '.join(map(str, selected_years))})")
            bar_fig = px.bar(
                tag_values.head(10),
                x=tag_values.head(10).index,
                y=tag_values.head(10).values,
                title=f'Top 10 Tags by {selected_metric} (Selected Years: {", ".join(map(str, selected_years))})',
                labels={'x': 'Tags', 'y': selected_metric},
                template='plotly_white'
            )
            st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.warning("Please select at least one year.")
    else:
        st.warning("The dataset does not contain 'Genres' or 'Year released' column.")
###########################################################################################

df = df[['AppID','Name','Release date', 'Genres','Categories','Tags','Developers','Publishers','About the game']]
df['AppID'] = df['AppID'].astype(str).sort_values().reset_index(drop=True)
df['Developers'] = df['Developers'].str.upper()
df['Publishers'] = df['Publishers'].str.upper()

df['Release date'] = pd.to_datetime(df['Release date'], format='%d/%m/%Y')
df['Month Release'] = df['Release date'].dt.strftime('01/%m/%Y')


if st.sidebar.checkbox("Genres"):
    st.title('Genres of Games Monthly')
    
    # Split and process genres
    df_melted_gen = df.assign(Genres=df['Genres'].str.split(',')).explode('Genres')
    df_melted_gen['Genres'] = df_melted_gen['Genres'].str.strip()
    
    # Convert Month Release
    df_melted_gen['Month Release'] = pd.to_datetime(df_melted_gen['Month Release'], format='%d/%m/%Y')
    df_melted_gen['Month-Year'] = df_melted_gen['Month Release'].dt.strftime('%m-%Y')
    
    # Get unique genres for sidebar selection
    unique_genres = sorted(df_melted_gen['Genres'].dropna().unique())
    selected_genre = st.sidebar.selectbox("Select Genre", unique_genres)
    
    # Filter data for selected genre
    genre_data = df_melted_gen[df_melted_gen['Genres'] == selected_genre]
    
    # Create monthly trend plot
    monthly_counts = genre_data.groupby('Month Release').size().reset_index()
    monthly_counts.columns = ['date', 'count']
    monthly_counts = monthly_counts.sort_values('date')
    monthly_counts['Month-Year'] = monthly_counts['date'].dt.strftime('%m-%Y')
    
    fig_gen = go.Figure(data=[
        go.Bar(
            x=monthly_counts['Month-Year'],
            y=monthly_counts['count'],
            name='Monthly Trend'
        )
    ])
    
    fig_gen.update_layout(
        title=f"Monthly Releases for {selected_genre}",
        xaxis_title="Month-Year",
        yaxis_title="Number of Games",
        height=400
    )
    
    # Seasonal Analysis
    # Map months to seasons
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    
    df_melted_gen['Season'] = df_melted_gen['Month Release'].dt.month.map(season_map)
    
    top_5_genres = df_melted_gen['Genres'].value_counts().head(5).index.tolist()
    
    # Filter for top 5 genres and get counts by season
    seasonal_data = df_melted_gen[df_melted_gen['Genres'].isin(top_5_genres)]
    season_genre_counts = seasonal_data.groupby(['Season', 'Genres']).size().reset_index(name='Count')
    
    df_melted_gen['Year'] = df_melted_gen['Month Release'].dt.year
    yearly_counts = df_melted_gen.groupby(['Year', 'Genres']).size().reset_index(name='Count')
    yearly_growth = yearly_counts.pivot(index='Year', columns='Genres', values='Count')
    growth_pct = yearly_growth.pct_change() * 100
    #st.write(yearly_counts)
    #Get average growth rate for each genre

    avg_growth = growth_pct.mean().sort_values(ascending=False).dropna()
    # Create figure
    fig = go.Figure()
    
    # Add bars for each genre
    for genre in top_5_genres:
        genre_data = season_genre_counts[season_genre_counts['Genres'] == genre]
        fig.add_trace(
            go.Bar(
                name=genre,
                x=genre_data['Season'],
                y=genre_data['Count']
            )
        )
    
    # Update layout
    fig.update_layout(
        barmode='group',
        title="Top 5 Genres Distribution Across Seasons",
        xaxis_title="Season_Release",
        yaxis_title="Number of Games",
        xaxis={'categoryorder':'array', 
               'categoryarray':['Spring', 'Summer', 'Fall', 'Winter']},
        height=500,
        showlegend=True,
        legend_title="Genres"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Growing Genres")
        st.bar_chart(avg_growth.head())
    with col2:
        st.subheader("Top 5 Declining Genres")
        st.bar_chart(avg_growth.tail(),color = '#ff0000')
    
    # Show plot_Seasonal Analysis
    st.subheader('Seasonal Analysis of All Genres')
    st.plotly_chart(fig, use_container_width=True)

    # Display monthly trend
    st.plotly_chart(fig_gen, use_container_width=True)

##### Developer and Publisher Analysis #####

    st.title('Top Developers and Publishers Analysis')
    
    # Process genres
    df_melted_gen = df.assign(Genres=df['Genres'].str.split(',')).explode('Genres')
    df_melted_gen['Genres'] = df_melted_gen['Genres'].str.strip()
    
    # Genre selection
    unique_genres = sorted(df_melted_gen['Genres'].dropna().unique())
    
    # Filter data for selected genre
    genre_data = df_melted_gen[df_melted_gen['Genres'] == selected_genre]
    
    col1, col2 = st.columns(2)
    
    with col1:
        #st.subheader(f'Top Developers for {selected_genre} Games')
        top_devs = genre_data["Developers"].value_counts().head(10)
        # Create bar chart for developers
        fig_dev = go.Figure(data=[
            go.Bar(
                x=top_devs.values,
                y=top_devs.index,
                orientation='h',
                marker_color='skyblue'
            )
        ])
        
        fig_dev.update_layout(
            title=f"Top 10 Developers in {selected_genre} Games",
            xaxis_title="Number of Games",
            yaxis_title="Developers",
            height=400,
            yaxis={'categoryorder':'total ascending'}
        )
        
        st.plotly_chart(fig_dev, use_container_width=True)

    with col2:
        #st.subheader(f'Top Publishers for {selected_genre} Games')
        # Get top publishers for selected genre
        top_pubs = genre_data["Publishers"].value_counts().head(10)
        
        # Create bar chart for publishers
        fig_pub = go.Figure(data=[
            go.Bar(
                x=top_pubs.values,
                y=top_pubs.index,
                orientation='h',
                marker_color='lightgreen'
            )
        ])
        
        fig_pub.update_layout(
            title=f"Top 10 Publishers in {selected_genre} Games",
            xaxis_title="Number of Games",
            yaxis_title="Publishers",
            height=400,
            yaxis={'categoryorder':'total ascending'}
        )
        
        st.plotly_chart(fig_pub, use_container_width=True)

## Word Cloude ##
    st.title('Word Cloud Analysis of Game Descriptions')
        
    # Add gaming-specific stop words
    df_melted_gen = df.assign(Genres=df['Genres'].str.split(',')).explode('Genres')
    df_melted_gen['Genres'] = df_melted_gen['Genres'].str.strip()
    
    # Genre selection
    unique_genres = sorted(df_melted_gen['Genres'].dropna().unique())
    # Filter data for selected genre
    genre_data = df_melted_gen[df_melted_gen['Genres'] == selected_genre]
    
    # Define stop words
    stop_words = {'game', 'games', 'play', 'player', 'players', 'feature', 'features', 
                 'will', 'can', 'the', 'and', 'for', 'that', 'you', 'with', 'are',
                 'your', 'this', 'from', 'has', 'have', 'all', 'get', 'more', 'one',
                 'new', 'also', 'about', 'out', 'who', 'but', 'its', "it's", 'was','not',
                 'what', 'which', 'when', 'where', 'why', 'how', 'they', 'their', 'them',
                 'there', 'then', 'than', 'these', 'those', 'other', 'some', 'many', 'much',
                 'most', 'such', 'each', 'every', 'own', 'just', 'still', 'even', 'both',
                 'either', 'neither', 'whether', 'either', 'or', 'if', 'so', 'as', 'at',
                 'by', 'on', 'in', 'to', 'into', 'up', 'down', 'over', 'under', 'through',
                 'between', 'among', 'after', 'before', 'above', 'below', 'behind', 'beside',
                 'near', 'off', 'on', 'out', 'around', 'throughout', 'upon', 'with', 'within',
                 'without', 'against', 'along', 'amid', 'among', 'beneath', 'beside', 'beyond',
                 'despite', 'during', 'inside', 'onto', 'outside', 'underneath', 'underneath',
                 'available', 'includes', 'including', 'different', 'various', 'like',
                 'make', 'made', 'way', 'well', 'use', 'using', 'used', 'etc', 'etc.',
                 'able', 'across', 'actually', 'almost', 'already', 'always', 'another',
                 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 
                 'areas', 'become', 'becomes', 'been', 'being', 'best', 'better',
                 'big', 'but', 'came', 'can', 'cannot', 'could', 'did',
                 'does', 'doing', 'done', 'each', 'find', 'first', 'good', 'great',
                 'had', 'now', 'see', 'want', 'while', 'via', 'way', 'get', 'going',
                 'included', 'including', 'features', 'feature', 'complete', 'based',
                 'his', 'her', 'him', 'she', 'he', 'they', 'them', 'their', 'our', 'us', 'we',
                 'i', 'you', 'your', 'my', 'mine', 'our', 'ours','theirs','only','every','any',
                 'few','little','lot','less','fewer'}
    
    # Text preprocessing function
    def preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Remove stopwords
        words = text.split()
        words = [w for w in words if w not in stop_words and len(w) > 2]
        return words
    
    # Combine all game descriptions for the selected genre
    text = ' '.join(genre_data['About the game'].dropna().astype(str))
    words = preprocess_text(text)
    
    # Get word frequencies
    word_freq = pd.Series(Counter(words)).sort_values(ascending=False)
    
    # Create bar chart for top 20 words
    fig_bar = go.Figure(data=[
        go.Bar(
            x=word_freq.head(20).index,
            y=word_freq.head(20).values,
            marker_color='steelblue'
        )
    ])
    
    fig_bar.update_layout(
        title=f"Top 20 Most Common Words in {selected_genre} Games",
        xaxis_title="Words",
        yaxis_title="Frequency",
        height=500,
        xaxis={'tickangle': 45}
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)

    st.title('Game Name Generator')
    # Process genres for genre selection
    df_melted_gen = df.assign(Genres=df['Genres'].str.split(',')).explode('Genres')
    df_melted_gen['Genres'] = df_melted_gen['Genres'].str.strip()
    
    # Genre selection
    unique_genres = sorted(df_melted_gen['Genres'].dropna().unique())
    
    # Filter data for selected genre
    genre_data = df_melted_gen[df_melted_gen['Genres'] == selected_genre]
    
    # Common word patterns for game names
    prefixes = ['The', 'Rise of', 'Age of', 'Legend of', 'Tales of', 'Chronicles of', 'World of']
    suffixes = ['Legacy', 'Chronicles', 'Saga', 'Adventures', 'Quest', 'Journey', 'Wars']
    
    # Get meaningful words from game descriptions
    text = ' '.join(genre_data['About the game'].dropna().astype(str))
    
    # Debug: Print sample of text
    #st.write("Debug - Sample of text:", text[:200])
    
    # Process text to get meaningful words
    words = re.findall(r'\b[A-Za-z]+\b', text.lower())
    word_freq = Counter(words)
    
    # Debug: Print word frequency
    #st.write("Debug - Number of unique words found:", len(word_freq))
    #st.write("Debug - Sample of word frequencies:", dict(list(word_freq.items())[:10]))
    
    # Define stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                 'by', 'from', 'up', 'about', 'into', 'over', 'after'}
    
    # Filter for words that might make good game names
    meaningful_words = [word for word, freq in word_freq.items() 
                       if len(word) > 4 
                       and freq < 100 
                       and freq > 2 
                       and word not in stop_words
                       and not any(c.isdigit() for c in word)]
    
    # Debug: Print meaningful words
    #st.write("Debug - Number of meaningful words:", len(meaningful_words))
    #st.write("Debug - Sample of meaningful words:", meaningful_words[:10])
    
    # Ensure we have meaningful words
    if not meaningful_words:
        st.error("No meaningful words found in the game descriptions. Adding default words.")
        meaningful_words = ['warrior', 'dragon', 'castle', 'knight', 'magic', 'sword', 'battle']
    
    # Generate game names
    st.subheader(f"Generated Game Names {selected_genre}")
    
    for i in range(10):
        name_pattern = random.choice([
            f"{random.choice(prefixes)} {random.choice(meaningful_words).title()}",
            f"{random.choice(meaningful_words).title()} {random.choice(suffixes)}",
            f"{random.choice(meaningful_words).title()} of {random.choice(meaningful_words).title()}",
            f"The {random.choice(meaningful_words).title()} {random.choice(meaningful_words).title()}"
        ])
        
        # Create a description based on the genre and words used
        if "of" in name_pattern:
            description = f"A {selected_genre.lower()} game where players explore the world of {name_pattern.split('of')[1].strip()}."
        else:
            word1 = random.choice(meaningful_words)
            word2 = random.choice(meaningful_words)
            description = f"An epic {selected_genre.lower()} adventure featuring {word1} and {word2}."
        
        # Display the name and description
        st.write(f"**{name_pattern}**")
        st.write(f"*{description}*")
        st.write("---")

################################################################################################################################

if st.sidebar.checkbox("Categories"):
    st.title('Categories of Games Monthly')
    
    # Split and process Categories
    df_melted_cat = df.assign(Categories=df['Categories'].str.split(',')).explode('Categories')
    df_melted_cat['Categories'] = df_melted_cat['Categories'].str.strip()
    
    # Convert Month Release
    df_melted_cat['Month Release'] = pd.to_datetime(df_melted_cat['Month Release'], format='%d/%m/%Y')
    df_melted_cat['Month-Year'] = df_melted_cat['Month Release'].dt.strftime('%m-%Y')
    
    # Get unique Categories for sidebar selection
    unique_Categories = sorted(df_melted_cat['Categories'].dropna().unique())
    selected_cat = st.sidebar.selectbox("Select Category", unique_Categories)
    
    # Filter data for selected cat
    cat_data = df_melted_cat[df_melted_cat['Categories'] == selected_cat]
    
    # Create monthly trend plot
    monthly_counts = cat_data.groupby('Month Release').size().reset_index()
    monthly_counts.columns = ['date', 'count']
    monthly_counts = monthly_counts.sort_values('date')
    monthly_counts['Month-Year'] = monthly_counts['date'].dt.strftime('%m-%Y')
    
    fig_gen = go.Figure(data=[
        go.Bar(
            x=monthly_counts['Month-Year'],
            y=monthly_counts['count'],
            name='Monthly Trend'
        )
    ])
    
    fig_gen.update_layout(
        title=f"Monthly Releases for {selected_cat}",
        xaxis_title="Month-Year",
        yaxis_title="Number of Games",
        height=400
    )
    
    # Seasonal Analysis
    # Map months to seasons
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    
    df_melted_cat['Season'] = df_melted_cat['Month Release'].dt.month.map(season_map)
    
    top_5_Categories = df_melted_cat['Categories'].value_counts().head(5).index.tolist()
    
    # Filter for top 5 Categories and get counts by season
    seasonal_data = df_melted_cat[df_melted_cat['Categories'].isin(top_5_Categories)]
    season_cat_counts = seasonal_data.groupby(['Season', 'Categories']).size().reset_index(name='Count')
    
    df_melted_cat['Year'] = df_melted_cat['Month Release'].dt.year
    yearly_counts = df_melted_cat.groupby(['Year', 'Categories']).size().reset_index(name='Count')
    yearly_growth = yearly_counts.pivot(index='Year', columns='Categories', values='Count')
    growth_pct = yearly_growth.pct_change() * 100
    #st.write(yearly_counts)
    #Get average growth rate for each cat

    avg_growth = growth_pct.mean().sort_values(ascending=False).dropna()
    # Create figure
    fig = go.Figure()
    
    # Add bars for each cat
    for cat in top_5_Categories:
        cat_data = season_cat_counts[season_cat_counts['Categories'] == cat]
        fig.add_trace(
            go.Bar(
                name=cat,
                x=cat_data['Season'],
                y=cat_data['Count']
            )
        )
    
    # Update layout
    fig.update_layout(
        barmode='group',
        title="Top 5 Categories Distribution Across Seasons",
        xaxis_title="Season_Release",
        yaxis_title="Number of Games",
        xaxis={'categoryorder':'array', 
               'categoryarray':['Spring', 'Summer', 'Fall', 'Winter']},
        height=500,
        showlegend=True,
        legend_title="Categories"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Growing Categories")
        st.bar_chart(avg_growth.head())
    with col2:
        st.subheader("Top 5 Declining Categories")
        st.bar_chart(avg_growth.tail(),color = '#ff0000')
    
    # Show plot_Seasonal Analysis
    st.subheader('Seasonal Analysis of All Categories')
    st.plotly_chart(fig, use_container_width=True)

    # Display monthly trend
    st.plotly_chart(fig_gen, use_container_width=True)

##### Developer and Publisher Analysis #####

    st.title('Top Developers and Publishers Analysis')
    
    # Process Categories
    df_melted_cat = df.assign(Categories=df['Categories'].str.split(',')).explode('Categories')
    df_melted_cat['Categories'] = df_melted_cat['Categories'].str.strip()
    
    # cat selection
    unique_Categories = sorted(df_melted_cat['Categories'].dropna().unique())
    
    # Filter data for selected cat
    cat_data = df_melted_cat[df_melted_cat['Categories'] == selected_cat]
    
    col1, col2 = st.columns(2)
    
    with col1:
        #st.subheader(f'Top Developers for {selected_cat} Games')
        top_devs = cat_data["Developers"].value_counts().head(10)
        # Create bar chart for developers
        fig_dev = go.Figure(data=[
            go.Bar(
                x=top_devs.values,
                y=top_devs.index,
                orientation='h',
                marker_color='skyblue'
            )
        ])
        
        fig_dev.update_layout(
            title=f"Top 10 Developers in {selected_cat} Games",
            xaxis_title="Number of Games",
            yaxis_title="Developers",
            height=400,
            yaxis={'categoryorder':'total ascending'}
        )
        
        st.plotly_chart(fig_dev, use_container_width=True)

    with col2:
        #st.subheader(f'Top Publishers for {selected_cat} Games')
        # Get top publishers for selected cat
        top_pubs = cat_data["Publishers"].value_counts().head(10)
        
        # Create bar chart for publishers
        fig_pub = go.Figure(data=[
            go.Bar(
                x=top_pubs.values,
                y=top_pubs.index,
                orientation='h',
                marker_color='lightgreen'
            )
        ])
        
        fig_pub.update_layout(
            title=f"Top 10 Publishers in {selected_cat} Games",
            xaxis_title="Number of Games",
            yaxis_title="Publishers",
            height=400,
            yaxis={'categoryorder':'total ascending'}
        )
        
        st.plotly_chart(fig_pub, use_container_width=True)

## Word Cloude ##
    st.title('Word Cloud Analysis of Game Descriptions')
        
    # Add gaming-specific stop words
    df_melted_cat = df.assign(Categories=df['Categories'].str.split(',')).explode('Categories')
    df_melted_cat['Categories'] = df_melted_cat['Categories'].str.strip()
    
    # cat selection
    unique_Categories = sorted(df_melted_cat['Categories'].dropna().unique())
    # Filter data for selected cat
    cat_data = df_melted_cat[df_melted_cat['Categories'] == selected_cat]
    
    # Define stop words
    stop_words = {'game', 'games', 'play', 'player', 'players', 'feature', 'features', 
                 'will', 'can', 'the', 'and', 'for', 'that', 'you', 'with', 'are',
                 'your', 'this', 'from', 'has', 'have', 'all', 'get', 'more', 'one',
                 'new', 'also', 'about', 'out', 'who', 'but', 'its', "it's", 'was','not',
                 'what', 'which', 'when', 'where', 'why', 'how', 'they', 'their', 'them',
                 'there', 'then', 'than', 'these', 'those', 'other', 'some', 'many', 'much',
                 'most', 'such', 'each', 'every', 'own', 'just', 'still', 'even', 'both',
                 'either', 'neither', 'whether', 'either', 'or', 'if', 'so', 'as', 'at',
                 'by', 'on', 'in', 'to', 'into', 'up', 'down', 'over', 'under', 'through',
                 'between', 'among', 'after', 'before', 'above', 'below', 'behind', 'beside',
                 'near', 'off', 'on', 'out', 'around', 'throughout', 'upon', 'with', 'within',
                 'without', 'against', 'along', 'amid', 'among', 'beneath', 'beside', 'beyond',
                 'despite', 'during', 'inside', 'onto', 'outside', 'underneath', 'underneath',
                 'available', 'includes', 'including', 'different', 'various', 'like',
                 'make', 'made', 'way', 'well', 'use', 'using', 'used', 'etc', 'etc.',
                 'able', 'across', 'actually', 'almost', 'already', 'always', 'another',
                 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 
                 'areas', 'become', 'becomes', 'been', 'being', 'best', 'better',
                 'big', 'but', 'came', 'can', 'cannot', 'could', 'did',
                 'does', 'doing', 'done', 'each', 'find', 'first', 'good', 'great',
                 'had', 'now', 'see', 'want', 'while', 'via', 'way', 'get', 'going',
                 'included', 'including', 'features', 'feature', 'complete', 'based',
                 'his', 'her', 'him', 'she', 'he', 'they', 'them', 'their', 'our', 'us', 'we',
                 'i', 'you', 'your', 'my', 'mine', 'our', 'ours','theirs','only','every','any',
                 'few','little','lot','less','fewer'}
    
    # Text preprocessing function
    def preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Remove stopwords
        words = text.split()
        words = [w for w in words if w not in stop_words and len(w) > 2]
        return words
    
    # Combine all game descriptions for the selected cat
    text = ' '.join(cat_data['About the game'].dropna().astype(str))
    words = preprocess_text(text)
    
    # Get word frequencies
    word_freq = pd.Series(Counter(words)).sort_values(ascending=False)
    
    # Create bar chart for top 20 words
    fig_bar = go.Figure(data=[
        go.Bar(
            x=word_freq.head(20).index,
            y=word_freq.head(20).values,
            marker_color='steelblue'
        )
    ])
    
    fig_bar.update_layout(
        title=f"Top 20 Most Common Words in {selected_cat} Games",
        xaxis_title="Words",
        yaxis_title="Frequency",
        height=500,
        xaxis={'tickangle': 45}
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)

    st.title('Game Name Generator')
    # Process Categories for cat selection
    df_melted_cat = df.assign(Categories=df['Categories'].str.split(',')).explode('Categories')
    df_melted_cat['Categories'] = df_melted_cat['Categories'].str.strip()
    
    # cat selection
    unique_Categories = sorted(df_melted_cat['Categories'].dropna().unique())
    
    # Filter data for selected cat
    cat_data = df_melted_cat[df_melted_cat['Categories'] == selected_cat]
    
    # Common word patterns for game names
    prefixes = ['The', 'Rise of', 'Age of', 'Legend of', 'Tales of', 'Chronicles of', 'World of']
    suffixes = ['Legacy', 'Chronicles', 'Saga', 'Adventures', 'Quest', 'Journey', 'Wars']
    
    # Get meaningful words from game descriptions
    text = ' '.join(cat_data['About the game'].dropna().astype(str))
    
    # Debug: Print sample of text
    #st.write("Debug - Sample of text:", text[:200])
    
    # Process text to get meaningful words
    words = re.findall(r'\b[A-Za-z]+\b', text.lower())
    word_freq = Counter(words)
    
    # Debug: Print word frequency
    #st.write("Debug - Number of unique words found:", len(word_freq))
    #st.write("Debug - Sample of word frequencies:", dict(list(word_freq.items())[:10]))
    
    # Define stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                 'by', 'from', 'up', 'about', 'into', 'over', 'after'}
    
    # Filter for words that might make good game names
    meaningful_words = [word for word, freq in word_freq.items() 
                       if len(word) > 4 
                       and freq < 100 
                       and freq > 2 
                       and word not in stop_words
                       and not any(c.isdigit() for c in word)]
    
    # Debug: Print meaningful words
    #st.write("Debug - Number of meaningful words:", len(meaningful_words))
    #st.write("Debug - Sample of meaningful words:", meaningful_words[:10])
    
    # Ensure we have meaningful words
    if not meaningful_words:
        st.error("No meaningful words found in the game descriptions. Adding default words.")
        meaningful_words = ['warrior', 'dragon', 'castle', 'knight', 'magic', 'sword', 'battle']
    
    # Generate game names
    st.subheader(f"Generated Game Names {selected_cat}")
    
    for i in range(10):
        name_pattern = random.choice([
            f"{random.choice(prefixes)} {random.choice(meaningful_words).title()}",
            f"{random.choice(meaningful_words).title()} {random.choice(suffixes)}",
            f"{random.choice(meaningful_words).title()} of {random.choice(meaningful_words).title()}",
            f"The {random.choice(meaningful_words).title()} {random.choice(meaningful_words).title()}"
        ])
        
        # Create a description based on the cat and words used
        if "of" in name_pattern:
            description = f"A {selected_cat.lower()} game where players explore the world of {name_pattern.split('of')[1].strip()}."
        else:
            word1 = random.choice(meaningful_words)
            word2 = random.choice(meaningful_words)
            description = f"An epic {selected_cat.lower()} adventure featuring {word1} and {word2}."
        
        # Display the name and description
        st.write(f"**{name_pattern}**")
        st.write(f"*{description}*")
        st.write("---")

################################################################################################################################

if st.sidebar.checkbox("Tags"):
    st.title('Tags of Games Monthly')
    
    # Split and process Tags
    df_melted_tags = df.assign(Tags=df['Tags'].str.split(',')).explode('Tags')
    df_melted_tags['Tags'] = df_melted_tags['Tags'].str.strip()
    
    # Convert Month Release
    df_melted_tags['Month Release'] = pd.to_datetime(df_melted_tags['Month Release'], format='%d/%m/%Y')
    df_melted_tags['Month-Year'] = df_melted_tags['Month Release'].dt.strftime('%m-%Y')
    
    # Get unique Tags for sidebar selection
    unique_Tags = sorted(df_melted_tags['Tags'].dropna().unique())
    selected_tags = st.sidebar.selectbox("Select Tags", unique_Tags)
    
    # Filter data for selected tags
    tags_data = df_melted_tags[df_melted_tags['Tags'] == selected_tags]
    
    # Create monthly trend plot
    monthly_counts = tags_data.groupby('Month Release').size().reset_index()
    monthly_counts.columns = ['date', 'count']
    monthly_counts = monthly_counts.sort_values('date')
    monthly_counts['Month-Year'] = monthly_counts['date'].dt.strftime('%m-%Y')
    
    fig_gen = go.Figure(data=[
        go.Bar(
            x=monthly_counts['Month-Year'],
            y=monthly_counts['count'],
            name='Monthly Trend'
        )
    ])
    
    fig_gen.update_layout(
        title=f"Monthly Releases for {selected_tags}",
        xaxis_title="Month-Year",
        yaxis_title="Number of Games",
        height=400
    )
    
    # Seasonal Analysis
    # Map months to seasons
    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    
    df_melted_tags['Season'] = df_melted_tags['Month Release'].dt.month.map(season_map)
    
    top_5_Tags = df_melted_tags['Tags'].value_counts().head(5).index.tolist()
    
    # Filter for top 5 Tags and get counts by season
    seasonal_data = df_melted_tags[df_melted_tags['Tags'].isin(top_5_Tags)]
    season_tags_counts = seasonal_data.groupby(['Season', 'Tags']).size().reset_index(name='Count')
    
    df_melted_tags['Year'] = df_melted_tags['Month Release'].dt.year
    yearly_counts = df_melted_tags.groupby(['Year', 'Tags']).size().reset_index(name='Count')
    yearly_growth = yearly_counts.pivot(index='Year', columns='Tags', values='Count')
    growth_pct = yearly_growth.pct_change() * 100
    st.write(yearly_counts)
    #Get average growth rate for each tags

    avg_growth = growth_pct.mean().sort_values(ascending=False).dropna()
    st.write(avg_growth)
    # Create figure
    fig = go.Figure()
    
    # Add bars for each tags
    for tags in top_5_Tags:
        tags_data = season_tags_counts[season_tags_counts['Tags'] == tags]
        fig.add_trace(
            go.Bar(
                name=tags,
                x=tags_data['Season'],
                y=tags_data['Count']
            )
        )
    
    # Update layout
    fig.update_layout(
        barmode='group',
        title="Top 5 Tags Distribution Across Seasons",
        xaxis_title="Season_Release",
        yaxis_title="Number of Games",
        xaxis={'categoryorder':'array', 
               'categoryarray':['Spring', 'Summer', 'Fall', 'Winter']},
        height=500,
        showlegend=True,
        legend_title="Tags"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Growing Tags")
        st.bar_chart(avg_growth.head())
    with col2:
        st.subheader("Top 5 Declining Tags")
        st.bar_chart(avg_growth.tail(),color = '#ff0000')
    
    # Show plot_Seasonal Analysis
    st.subheader('Seasonal Analysis of All Tags')
    st.plotly_chart(fig, use_container_width=True)

    # Display monthly trend
    st.plotly_chart(fig_gen, use_container_width=True)

##### Developer and Publisher Analysis #####

    st.title('Top Developers and Publishers Analysis')
    
    # Process Tags
    df_melted_tags = df.assign(Tags=df['Tags'].str.split(',')).explode('Tags')
    df_melted_tags['Tags'] = df_melted_tags['Tags'].str.strip()
    
    # tags selection
    unique_Tags = sorted(df_melted_tags['Tags'].dropna().unique())
    
    # Filter data for selected tags
    tags_data = df_melted_tags[df_melted_tags['Tags'] == selected_tags]
    
    col1, col2 = st.columns(2)
    
    with col1:
        #st.subheader(f'Top Developers for {selected_tags} Games')
        top_devs = tags_data["Developers"].value_counts().head(10)
        # Create bar chart for developers
        fig_dev = go.Figure(data=[
            go.Bar(
                x=top_devs.values,
                y=top_devs.index,
                orientation='h',
                marker_color='skyblue'
            )
        ])
        
        fig_dev.update_layout(
            title=f"Top 10 Developers in {selected_tags} Games",
            xaxis_title="Number of Games",
            yaxis_title="Developers",
            height=400,
            yaxis={'categoryorder':'total ascending'}
        )
        
        st.plotly_chart(fig_dev, use_container_width=True)

    with col2:
        #st.subheader(f'Top Publishers for {selected_tags} Games')
        # Get top publishers for selected tags
        top_pubs = tags_data["Publishers"].value_counts().head(10)
        
        # Create bar chart for publishers
        fig_pub = go.Figure(data=[
            go.Bar(
                x=top_pubs.values,
                y=top_pubs.index,
                orientation='h',
                marker_color='lightgreen'
            )
        ])
        
        fig_pub.update_layout(
            title=f"Top 10 Publishers in {selected_tags} Games",
            xaxis_title="Number of Games",
            yaxis_title="Publishers",
            height=400,
            yaxis={'categoryorder':'total ascending'}
        )
        
        st.plotly_chart(fig_pub, use_container_width=True)

## Word Cloude ##
    st.title('Word Cloud Analysis of Game Descriptions')
        
    # Add gaming-specific stop words
    df_melted_tags = df.assign(Tags=df['Tags'].str.split(',')).explode('Tags')
    df_melted_tags['Tags'] = df_melted_tags['Tags'].str.strip()
    
    # tags selection
    unique_Tags = sorted(df_melted_tags['Tags'].dropna().unique())
    # Filter data for selected tags
    tags_data = df_melted_tags[df_melted_tags['Tags'] == selected_tags]
    
    # Define stop words
    stop_words = {'game', 'games', 'play', 'player', 'players', 'feature', 'features', 
                 'will', 'can', 'the', 'and', 'for', 'that', 'you', 'with', 'are',
                 'your', 'this', 'from', 'has', 'have', 'all', 'get', 'more', 'one',
                 'new', 'also', 'about', 'out', 'who', 'but', 'its', "it's", 'was','not',
                 'what', 'which', 'when', 'where', 'why', 'how', 'they', 'their', 'them',
                 'there', 'then', 'than', 'these', 'those', 'other', 'some', 'many', 'much',
                 'most', 'such', 'each', 'every', 'own', 'just', 'still', 'even', 'both',
                 'either', 'neither', 'whether', 'either', 'or', 'if', 'so', 'as', 'at',
                 'by', 'on', 'in', 'to', 'into', 'up', 'down', 'over', 'under', 'through',
                 'between', 'among', 'after', 'before', 'above', 'below', 'behind', 'beside',
                 'near', 'off', 'on', 'out', 'around', 'throughout', 'upon', 'with', 'within',
                 'without', 'against', 'along', 'amid', 'among', 'beneath', 'beside', 'beyond',
                 'despite', 'during', 'inside', 'onto', 'outside', 'underneath', 'underneath',
                 'available', 'includes', 'including', 'different', 'various', 'like',
                 'make', 'made', 'way', 'well', 'use', 'using', 'used', 'etc', 'etc.',
                 'able', 'across', 'actually', 'almost', 'already', 'always', 'another',
                 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 
                 'areas', 'become', 'becomes', 'been', 'being', 'best', 'better',
                 'big', 'but', 'came', 'can', 'cannot', 'could', 'did',
                 'does', 'doing', 'done', 'each', 'find', 'first', 'good', 'great',
                 'had', 'now', 'see', 'want', 'while', 'via', 'way', 'get', 'going',
                 'included', 'including', 'features', 'feature', 'complete', 'based',
                 'his', 'her', 'him', 'she', 'he', 'they', 'them', 'their', 'our', 'us', 'we',
                 'i', 'you', 'your', 'my', 'mine', 'our', 'ours','theirs','only','every','any',
                 'few','little','lot','less','fewer'}
    
    # Text preprocessing function
    def preprocess_text(text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Remove stopwords
        words = text.split()
        words = [w for w in words if w not in stop_words and len(w) > 2]
        return words
    
    # Combine all game descriptions for the selected tags
    text = ' '.join(tags_data['About the game'].dropna().astype(str))
    words = preprocess_text(text)
    
    # Get word frequencies
    word_freq = pd.Series(Counter(words)).sort_values(ascending=False)
    
    # Create bar chart for top 20 words
    fig_bar = go.Figure(data=[
        go.Bar(
            x=word_freq.head(20).index,
            y=word_freq.head(20).values,
            marker_color='steelblue'
        )
    ])
    
    fig_bar.update_layout(
        title=f"Top 20 Most Common Words in {selected_tags} Games",
        xaxis_title="Words",
        yaxis_title="Frequency",
        height=500,
        xaxis={'tickangle': 45}
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)

    st.title('Game Name Generator')
    # Process Tags for tags selection
    df_melted_tags = df.assign(Tags=df['Tags'].str.split(',')).explode('Tags')
    df_melted_tags['Tags'] = df_melted_tags['Tags'].str.strip()
    
    # tags selection
    unique_Tags = sorted(df_melted_tags['Tags'].dropna().unique())
    
    # Filter data for selected tags
    tags_data = df_melted_tags[df_melted_tags['Tags'] == selected_tags]
    
    # Common word patterns for game names
    prefixes = ['The', 'Rise of', 'Age of', 'Legend of', 'Tales of', 'Chronicles of', 'World of']
    suffixes = ['Legacy', 'Chronicles', 'Saga', 'Adventures', 'Quest', 'Journey', 'Wars']
    
    # Get meaningful words from game descriptions
    text = ' '.join(tags_data['About the game'].dropna().astype(str))
    
    # Debug: Print sample of text
    #st.write("Debug - Sample of text:", text[:200])
    
    # Process text to get meaningful words
    words = re.findall(r'\b[A-Za-z]+\b', text.lower())
    word_freq = Counter(words)
    
    # Debug: Print word frequency
    #st.write("Debug - Number of unique words found:", len(word_freq))
    #st.write("Debug - Sample of word frequencies:", dict(list(word_freq.items())[:10]))
    
    # Define stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                 'by', 'from', 'up', 'about', 'into', 'over', 'after'}
    
    # Filter for words that might make good game names
    meaningful_words = [word for word, freq in word_freq.items() 
                       if len(word) > 4 
                       and freq < 100 
                       and freq > 2 
                       and word not in stop_words
                       and not any(c.isdigit() for c in word)]
    
    # Debug: Print meaningful words
    #st.write("Debug - Number of meaningful words:", len(meaningful_words))
    #st.write("Debug - Sample of meaningful words:", meaningful_words[:10])
    
    # Ensure we have meaningful words
    if not meaningful_words:
        st.error("No meaningful words found in the game descriptions. Adding default words.")
        meaningful_words = ['warrior', 'dragon', 'castle', 'knight', 'magic', 'sword', 'battle']
    
    # Generate game names
    st.subheader(f"Generated Game Names {selected_tags}")
    
    for i in range(10):
        name_pattern = random.choice([
            f"{random.choice(prefixes)} {random.choice(meaningful_words).title()}",
            f"{random.choice(meaningful_words).title()} {random.choice(suffixes)}",
            f"{random.choice(meaningful_words).title()} of {random.choice(meaningful_words).title()}",
            f"The {random.choice(meaningful_words).title()} {random.choice(meaningful_words).title()}"
        ])
        
        # Create a description based on the tags and words used
        if "of" in name_pattern:
            description = f"A {selected_tags.lower()} game where players explore the world of {name_pattern.split('of')[1].strip()}."
        else:
            word1 = random.choice(meaningful_words)
            word2 = random.choice(meaningful_words)
            description = f"An epic {selected_tags.lower()} adventure featuring {word1} and {word2}."
        
        # Display the name and description
        st.write(f"**{name_pattern}**")
        st.write(f"*{description}*")
        st.write("---")
