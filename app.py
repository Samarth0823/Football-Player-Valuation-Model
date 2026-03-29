import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# 1. APP CONFIG
st.set_page_config(page_title="FC Scout: AI Similarity Tool", layout="wide")

st.title("⚽ FC Scout: AI Similarity & Valuation Tool")
st.markdown("Find the next superstar by matching performance profiles.")

# 2. LOAD DATA
@st.cache_data
def load_data():
    # Make sure this filename matches exactly what's on your GitHub
    df = pd.read_csv('final_scout_data.csv')
    return df

try:
    df = load_data()
    
    # 3. DYNAMIC COLUMN CHECK
    # We use these features for similarity. We must ensure they exist in the CSV.
    sim_cols = ['MP', 'Gls', 'Ast', 'Age', 'G_per_MP', 'A_per_MP', 'GA_per_MP', 'is_top_5', 'young_producer']
    
    # Filter only for rows that have all required features
    df_clean = df.dropna(subset=sim_cols)

    # 4. SIDEBAR SEARCH
    st.sidebar.header("Scouting Filters")
    # Get unique player names sorted by market value
    player_options = df_clean.sort_values(by='market_value_in_eur', ascending=False)['Player'].unique()
    target_player = st.sidebar.selectbox("Select a High-Value Player to Replace:", player_options)

    # 5. SIMILARITY ENGINE
    # We fit the scaler on the performance metrics
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[sim_cols])
    
    def get_recommendations(name, top_n=5):
        # Find index of target player
        target_idx = df_clean[df_clean['Player'] == name].index[0]
        # Get position in the matrix
        matrix_idx = df_clean.index.get_loc(target_idx)
        
        target_vec = X_scaled[matrix_idx].reshape(1, -1)
        sim_scores = cosine_similarity(target_vector=target_vec, Y=X_scaled).flatten()
        
        # Get top indices (excluding the player themselves)
        similar_indices = sim_scores.argsort()[-(top_n+1):-1][::-1]
        
        return df_clean.iloc[similar_indices], sim_scores[similar_indices]

    # 6. DISPLAY RESULTS
    if st.button("Generate Scout Report"):
        results_df, scores = get_recommendations(target_player)
        
        if not results_df.empty:
            st.subheader(f"Top 5 Alternatives for {target_player}")
            
            for i, (idx, row) in enumerate(results_df.iterrows()):
                # Format the card
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    with col1:
                        st.markdown(f"### {i+1}. {row['Player']}")
                        st.caption(f"{row['Squad']} | {row.get('Comp', 'Unknown League')}")
                    with col2:
                        st.metric("Similarity", f"{scores[i]*100:.1f}%")
                    with col3:
                        st.metric("Market Value", f"€{row['market_value_in_eur']:,.0f}")
                    with col4:
                        st.metric("Age", int(row['Age']))
                    st.divider()
        else:
            st.warning("No similar players found. Please check your data filters.")

except Exception as e:
    st.error(f"⚠️ App Error: {e}")
    st.write("Current Columns in Data:", df.columns.tolist() if 'df' in locals() else "Data not loaded")
    st.info("Check if 'scout_master_data.csv' is uploaded and contains performance columns.")
