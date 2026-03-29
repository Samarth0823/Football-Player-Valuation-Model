import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# 1. APP CONFIG
st.set_page_config(page_title="FC Scout: AI Similarity Tool", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 10px; }
    </style>
    """, unsafe_base_content=True)

st.title("⚽ FC Scout: AI Similarity & Valuation Tool")
st.markdown("---")

@st.cache_data
def load_data():
    # Make sure this filename matches your GitHub exactly
    return pd.read_csv('final_scout_data.csv')

try:
    df = load_data()
    
    # Columns used for similarity math
    sim_cols = ['MP', 'Gls', 'Ast', 'Age', 'G_per_MP', 'A_per_MP', 'GA_per_MP', 'is_top_5', 'young_producer']
    
    # Check for missing columns
    missing_cols = [c for c in sim_cols + ['market_value_in_eur', 'Player', 'Squad'] if c not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing columns in CSV: {missing_cols}")
    else:
        # SIDEBAR
        st.sidebar.header("Scouting Filters")
        player_list = df.sort_values(by='market_value_in_eur', ascending=False)['Player'].unique()
        target_player = st.sidebar.selectbox("Select a Star Player to Replace:", player_list)
        
        # SLIDER FOR BUDGET
        budget_limit = st.sidebar.slider("Max Budget (Millions €)", 0, 200, 200)

        # 2. SIMILARITY LOGIC
        scaler = StandardScaler()
        # Scale only the performance stats
        X_scaled = scaler.fit_transform(df[sim_cols])
        
        if st.button("Generate Comprehensive Scout Report"):
            # Get target player info
            target_idx = df[df['Player'] == target_player].index[0]
            target_row = df.iloc[target_idx]
            target_vec = X_scaled[target_idx].reshape(1, -1)
            
            # Calculate Similarity
            sim_scores = cosine_similarity(target_vec, X_scaled).flatten()
            
            # Filter by budget and exclude the target player
            df['temp_sim'] = sim_scores
            recommendations = df[
                (df['Player'] != target_player) & 
                (df['market_value_in_eur'] <= budget_limit * 1000000)
            ].sort_values(by='temp_sim', ascending=False).head(5)

            st.subheader(f"Top 5 Strategic Alternatives for {target_player}")
            
            # 3. DISPLAY RESULTS
            for i, (r_idx, row) in enumerate(recommendations.iterrows()):
                similarity_pct = row['temp_sim'] * 100
                
                with st.container():
                    # Header Row
                    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
                    c1.markdown(f"### {i+1}. {row['Player']}")
                    c1.caption(f"{row['Squad']} | {row.get('Comp', 'League')}")
                    c2.metric("Similarity", f"{similarity_pct:.1f}%")
                    c3.metric("Market Value", f"€{row['market_value_in_eur']:,.0f}")
                    c4.metric("Age", int(row['Age']))

                    # Comparison Row
                    col_info, col_chart = st.columns([1, 1])
                    
                    with col_info:
                        st.write("**Why this match?**")
                        # Logic to find the most similar stat
                        diffs = np.abs(X_scaled[target_idx] - X_scaled[r_idx])
                        best_match_stat = sim_cols[np.argmin(diffs)]
                        st.info(f"💡 This player perfectly mirrors {target_player}'s output in **{best_match_stat}**.")
                        
                        # Bargain Alert
                        if row['market_value_in_eur'] < (target_row['market_value_in_eur'] * 0.5):
                            st.success("💎 **BARGAIN ALERT**: High similarity at less than 50% of the cost.")
                    
                    with col_chart:
                        # RADAR CHART
                        categories = ['MP', 'Gls', 'Ast', 'Age', 'GA_MP', 'Top5']
                        # Simplified categories for cleaner radar
                        target_vals = [target_row['MP'], target_row['Gls'], target_row['Ast'], 
                                       target_row['Age'], target_row['GA_per_MP'], target_row['is_top_5']]
                        row_vals = [row['MP'], row['Gls'], row['Ast'], 
                                    row['Age'], row['GA_per_MP'], row['is_top_5']]
                        
                        # Normalize values for radar (0 to 1)
                        target_norm = [v / max(target_vals + row_vals + [1]) for v in target_vals]
                        row_norm = [v / max(target_vals + row_vals + [1]) for v in row_vals]

                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(r=target_norm, theta=categories, fill='toself', name=target_player, line_color='#ef4444'))
                        fig.add_trace(go.Scatterpolar(r=row_norm, theta=categories, fill='toself', name=row['Player'], line_color='#3b82f6'))
                        
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=False)),
                            showlegend=True, height=300, margin=dict(l=40, r=40, t=20, b=20),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()

except Exception as e:
    st.error(f"General Error: {e}")
