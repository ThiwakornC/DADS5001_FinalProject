import streamlit as st
import pandas as pd
import base64
from PIL import Image
import requests
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Welcome Game Developer",
    page_icon="🎮",
)

st.write("# Welcome Game Developer 🎮")

st.write('# The Game Awards 2024 Winners')

col1, col2, col3 = st.columns(3)

with col2:
   st.image("https://cdn.thegameawards.com/1/2024/11/astro_bot.jpg", caption="ASTRO BOT 👑", width=300)

col4, col5, col6 , col7, col8= st.columns(5)

with col4:
   st.image("https://cdn.thegameawards.com/1/2024/11/balatro.jpg", caption="BALATRO", width=150)
with col5:
   st.image("https://cdn.thegameawards.com/1/2024/11/elden_ring.jpg", caption="ELDEN RING", width=150) 
with col6:
   st.image("https://cdn.thegameawards.com/1/2024/11/black_myth.jpg", caption="BLACK MYTH: WUKONG", width=150)

with col7:
   st.image("https://cdn.thegameawards.com/1/2024/11/ff7.jpg", caption="FINAL FANTASY VII REBIRTH", width=150)
with col8:
   st.image("https://cdn.thegameawards.com/1/2024/11/metaphor.jpg", caption="METAPHOR: REFANTAZIO", width=150)

@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\Thiwa\\dads5001\\Tools_Project\Game.csv", encoding='utf-8-sig') #change path
    return df
df = load_data()
######################################################################################

st.title('Genres of Games')
df_Gen = df[['AppID', 'Genres']]
df_Gen['AppID'] = df_Gen['AppID'].astype(str).sort_values().reset_index(drop=True)
df_melted_gen = df.assign(Genres=df['Genres'].str.split(',')).explode('Genres')
genre_counts_gen = df_melted_gen['Genres'].value_counts().sort_values(ascending=False)
st.bar_chart(genre_counts_gen)
#st.write(df_Gen)

######################################################################################

st.title('Categories of Games')
df_Cat = df[['AppID', 'Categories']]
df_Cat['AppID'] = df_Cat['AppID'].astype(str).sort_values().reset_index(drop=True)
df_melted_cat = df.assign(Categories=df['Categories'].str.split(',')).explode('Categories')
genre_counts_cat = df_melted_cat['Categories'].value_counts().sort_values(ascending=False)
st.bar_chart(genre_counts_cat)

######################################################################################

st.title('Tags of Games Top 20')
df_Tags = df[['AppID', 'Tags']]
df_Tags['AppID'] = df_Tags['AppID'].astype(str).sort_values().reset_index(drop=True)
df_melted_Tags = df.assign(Tags=df['Tags'].str.split(',')).explode('Tags')
genre_counts_Tags = df_melted_Tags['Tags'].value_counts().sort_values(ascending=False)
top_20_tags = genre_counts_Tags.head(20).sort_values(ascending=True)
st.bar_chart(top_20_tags)

#fig = px.bar(x=top_20_tags.index, y=top_20_tags.values,
#            title='Top 20 Game Tags',
#            labels={'x': 'Tags', 'y': 'Count'})
#
#fig.update_layout(
#   title_x=0.5,
#   xaxis_tickangle=-45,
#   height=500
#)
#
#fig.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
#
#st.plotly_chart(fig)

st.title('Most Games Review Top 10')
df_rw = df[['Name', 'Total number of reviews', 'Positive', 'Negative']]
top_10_rw = df_rw.sort_values('Total number of reviews', ascending=False).head(10)

# Calculate proportions
top_10_rw['Prop_Positive'] = top_10_rw['Positive'] / top_10_rw['Total number of reviews']
top_10_rw['Prop_Negative'] = top_10_rw['Negative'] / top_10_rw['Total number of reviews']

# Create two traces for negative and positive proportions
fig = go.Figure()

fig.add_trace(go.Bar(
    x=-top_10_rw['Prop_Negative'],
    y=top_10_rw['Name'],
    orientation='h',
    name='Negative',
    marker_color='red',
    text=[f"{x:,.0f}" for x in top_10_rw['Negative']],
    textposition='auto',
))

fig.add_trace(go.Bar(
    x=top_10_rw['Prop_Positive'],
    y=top_10_rw['Name'],
    orientation='h',
    name='Positive',
    marker_color='blue',
    text=[f"{x:,.0f}" for x in top_10_rw['Positive']],
    textposition='auto',
))

fig.update_layout(
    #title={
    #    'text': 'Most Review Top 10',
    #    'font': {'size': 30, 'color': '#3f51b5'},
    #    'x': 0.5,
    #    'xanchor': 'center'
    #},
    barmode='overlay',
    bargap=0.1,
    plot_bgcolor='#F0F4F9',
    paper_bgcolor='#F0F4F9',
    #height=800,  # เพิ่มความสูง
    width=1400,  # เพิ่มความกว้าง
    xaxis=dict(
        range=[-1, 1],
        tickformat='.1f',
        title='Proportion of Reviews',
        zeroline=True,
        zerolinecolor='black',
        zerolinewidth=1,
        showgrid=True,
        gridcolor='white',
        gridwidth=2,
        tickfont={'size': 14}  # เพิ่มขนาดตัวเลข
    ),
    yaxis=dict(
        title='',
        zeroline=False,
        showgrid=True,
        gridcolor='white',
        gridwidth=2,
        tickfont={'size': 14}  # เพิ่มขนาดชื่อเกม
    ),
    legend=dict(
        font=dict(size=14)  # เพิ่มขนาด legend
    ),
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig, use_container_width=True)

########################################################

st.title('Top 5 Positive & Negative Games Review')
df_p_n_rw = df[['Name','Positive', 'Negative']]

top_5_prw = df_p_n_rw.sort_values('Positive', ascending=False).head(5)
top_5_nrw = df_p_n_rw.sort_values('Negative', ascending=False).head(5)


fig_p = px.bar(top_5_prw, 
         x="Positive", 
         y="Name",
         template="plotly_white",
         color_discrete_sequence=['#1f77b4'],
         width=600)
fig_p.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
st.plotly_chart(fig_p, use_container_width=True)


fig_n = px.bar(top_5_nrw, 
         x="Negative", 
         y="Name", 
         template="plotly_white",
         color_discrete_sequence=['#ff0000'],
         width=600)
fig_n.update_traces(texttemplate='%{x:,.0f}', textposition='outside')
st.plotly_chart(fig_n, use_container_width=True)
