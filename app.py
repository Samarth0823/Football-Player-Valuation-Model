import streamlit as st
import pandas as pd
import numpy as np
try:
    import plotly.graph_objects as go
except ImportError:
    st.error("Please add 'plotly' to your requirements.txt file on GitHub!")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="FC Scout Pro", layout="wide")

# This forces a UI change you can see immediately
st.title("🚀 FC Scout: Advanced AI Dashboard") 

@st.cache_data
def load_data():
    # ENSURE THIS FILENAME IS CORRECT
    return pd.read_csv('final_scout_data.csv')

df = load_data()

# TEST: Does the data have the right columns?
sim_cols = ['MP', 'Gls', 'Ast', 'Age', 'G_per_MP', 'A_per_MP', 'GA_per_MP', 'is_top_5', 'young_producer']

# SIDEBAR
target_player = st.sidebar.selectbox("Select Player:", df['Player'].unique())

if st.button("Run Deep Dive Analysis"):
    # 1. Math
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[sim_cols].fillna(0))
    idx = df[df['Player'] == target_player].index[0]
    sim_scores = cosine_similarity(X_scaled[idx].reshape(1, -1), X_scaled).flatten()
    
    df['sim'] = sim_scores
    recs = df[df['Player'] != target_player].sort_values('sim', ascending=False).head(3)
    
    # 2. Advanced UI
    for i, (r_idx, row) in enumerate(recs.iterrows()):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.header(f"{i+1}. {row['Player']}")
            st.metric("Similarity Score", f"{row['sim']*100:.1f}%")
            st.metric("Value", f"€{row['market_value_in_eur']:,.0f}")
        with col2:
            # RADAR CHART
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=df.loc[idx, sim_cols].values, theta=sim_cols, fill='toself', name=target_player))
            fig.add_trace(go.Scatterpolar(r=row[sim_cols].values, theta=sim_cols, fill='toself', name=row['Player']))
            st.plotly_chart(fig)
        st.divider()
