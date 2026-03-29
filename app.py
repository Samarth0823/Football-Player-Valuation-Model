import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="FC Scout: AI Similarity Tool", layout="wide")
st.title("⚽ FC Scout: AI Similarity & Valuation Tool")

@st.cache_data
def load_data():
    return pd.read_csv('final_scout_data.csv')

try:
    df = load_data()
    
    # Required columns for the math to work
    sim_cols = ['MP', 'Gls', 'Ast', 'Age', 'G_per_MP', 'A_per_MP', 'GA_per_MP', 'is_top_5', 'young_producer']
    
    # Check if all columns exist in the CSV
    missing_cols = [c for c in sim_cols + ['market_value_in_eur', 'Player'] if c not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing columns in CSV: {missing_cols}")
    else:
        # Sidebar
        player_list = df.sort_values(by='market_value_in_eur', ascending=False)['Player'].unique()
        target_player = st.sidebar.selectbox("Select Player:", player_list)

        # Similarity Logic
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[sim_cols])
        
        if st.button("Generate Scout Report"):
            idx = df[df['Player'] == target_player].index[0]
            target_vec = X_scaled[idx].reshape(1, -1)
            sim_scores = cosine_similarity(target_vec, X_scaled).flatten()
            
            # Get top 5 (excluding self)
            sim_indices = sim_scores.argsort()[-6:-1][::-1]
            results = df.iloc[sim_indices]
            
            st.subheader(f"Top 5 Alternatives for {target_player}")
            for i, (r_idx, row) in enumerate(results.iterrows()):
                with st.container():
                    c1, c2, c3 = st.columns([2, 1, 1])
                    c1.write(f"### {i+1}. {row['Player']} ({row['Squad']})")
                    c2.metric("Market Value", f"€{row['market_value_in_eur']:,.0f}")
                    c3.metric("Similarity", f"{sim_scores[sim_indices[i]]*100:.1f}%")
                    st.divider()

except Exception as e:
    st.error(f"General Error: {e}")
